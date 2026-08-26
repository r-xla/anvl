#' @include array.R
#' @include box.R

#' @title Graph Value
#' @description
#' Value in an [`AnvlGraph`]. This is a mutable class.
#' @param aval ([`AbstractArray`])\cr
#'   The abstract value of the variable.
#' @return (`GraphValue`)
#' @export
GraphValue <- function(aval) {
  # hot-path constructor: no input validation
  env <- new.env(parent = emptyenv())
  env$aval <- aval

  structure(env, class = "GraphValue")
}

#' @title Graph Literal
#' @description
#' Literal in an [`AnvlGraph`]. This is a mutable class.
#' @param aval ([`LiteralArray`])\cr
#'   The value of the literal.
#' @return (`GraphLiteral`)
#' @export
GraphLiteral <- function(aval) {
  # hot-path constructor: no input validation
  env <- new.env(parent = emptyenv())
  env$aval <- aval

  structure(env, class = "GraphLiteral")
}

is_graph_literal <- function(x) {
  inherits(x, "GraphLiteral")
}

#' @title Graph R Data
#' @description
#' Node of an [`AnvlGraph`] standing for an R value whose data type is not
#' decided yet -- see [`RDataArray`]. It is not part of any primitive call:
#' it is *materialized* first, into a [`GraphLiteral`] (an in-body scalar), a
#' constant (an in-body R array), or a [`GraphValue`] input (an argument of the
#' jitted function), at the dtype the use site asks for. This is a mutable
#' class.
#' @param aval ([`RDataArray`])\cr
#'   The R value.
#' @return (`GraphRData`)
#' @seealso [RDataArray]
#' @export
GraphRData <- function(aval) {
  env <- new.env(parent = emptyenv())
  env$aval <- aval
  # dtype name -> the GraphBox this value was materialized into for it. One
  # entry per dtype the trace asked for, so asking twice reuses the value
  # rather than building it again.
  env$mat <- list()

  structure(env, class = "GraphRData")
}

is_graph_rdata <- function(x) {
  inherits(x, "GraphRData")
}

#' @export
format.GraphRData <- function(x, ...) {
  sprintf("GraphRData(%s)", format(x$aval))
}

#' @export
print.GraphRData <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
shape.GraphRData <- function(x, ...) {
  shape(x$aval)
}

#' @export
dtype.GraphRData <- function(x, ...) {
  dtype(x$aval)
}

is_rdata_box <- function(x) {
  inherits(x, "GraphBox") && inherits(x$gnode, "GraphRData")
}

# The dtype an uncommitted R value would take if nothing claimed it.
rdata_default_dtype <- function(box) {
  box$gnode$aval$default_dtype
}

# The dtype that holds an R value of this storage type exactly. Building at it
# is always faithful; every other dtype is reached by converting from it, so
# that narrowing follows the program's own conversion semantics rather than R's
# (they disagree: XLA clamps a float that overflows an integer dtype and maps
# NaN to zero, R wraps or yields NA).
rdata_natural_dtype <- function(r_type) {
  switch(r_type, double = as_dtype("f64"), integer = as_dtype("i32"), logical = as_dtype("bool"))
}

# Whether an R value of this storage type can be built at `dtype` directly,
# i.e. whether doing so gives the same value as building it at its natural
# dtype and converting. It does within the value's own category (float to
# float rounds once either way; an R integer reaches any 32-bit-or-wider
# integer dtype unchanged), and for an R integer widened into a float. It does
# not for anything narrowing out of a category, which is where R and the
# program disagree.
rdata_builds_directly <- function(r_type, dtype) {
  switch(
    r_type,
    logical = is_dtype_bool(dtype),
    integer = is_dtype_float(dtype) || (is_dtype_intish(dtype) && dtype_width(dtype) >= 32L),
    double = is_dtype_float(dtype)
  )
}

# Build `box`'s R value into the graph at `dtype`, and return the GraphBox for
# it. Memoized per dtype on the node: a value used twice at one dtype is built
# once. The value is always built from the R data itself, never converted from
# a value that was built at another dtype in the same category -- that is what
# keeps `x_f64 / sqrt(2)` exact.
materialize_rdata <- function(box, dtype) {
  node <- box$gnode
  key <- as.character(dtype)
  hit <- node$mat[[key]]
  if (!is.null(hit)) {
    return(hit)
  }
  aval <- node$aval
  if (!rdata_builds_directly(aval$r_type, dtype)) {
    # Out of the value's own category: build it where it is exact and let the
    # program convert, so the result is the one `nv_convert()` gives for a
    # typed array of the same value.
    out <- prim_convert(materialize_rdata(box, rdata_natural_dtype(aval$r_type)), dtype = dtype)
    node$mat[[key]] <- out
    return(out)
  }
  desc <- box$desc
  out <- if (is.null(aval$data)) {
    # An argument of the jitted function: its value is unknown here (the
    # compiled program must not depend on it), so it becomes an input of this
    # dtype and the call uploads the R data at that dtype.
    # finalize_rdata_inputs() puts it in the input list.
    register_gval(desc, GraphValue(AbstractArray(dtype = dtype, shape = aval$shape)))
  } else if (is.null(dim(aval$data))) {
    get_box_or_register_const(
      desc,
      GraphLiteral(LiteralArray(aval$data, shape = aval$shape, dtype = dtype))
    )
  } else {
    # An R array becomes a constant of the graph, built at the dtype the
    # program uses it at.
    get_box_or_register_const(desc, nv_array(aval$data, dtype = dtype))
  }
  node$mat[[key]] <- out
  out
}

# Materialize at the dtype the value commits to when nothing else decided.
commit_rdata <- function(box) {
  materialize_rdata(box, rdata_default_dtype(box))
}

#' @title Peek at a Data Type
#' @description
#' The data type `x` would use if an operation needed one right now: its own
#' where it has one, and the one it *would* commit to where it has none.
#'
#' An R value entering a program has no data type until it is used (see
#' [`RDataArray`]), so [`dtype()`][tengen::dtype] has nothing to report for one
#' and errors. `peek_dtype()` answers instead with the value's default (`f32`
#' for a double, `i32` for an integer, `bool` for a logical) -- *without*
#' committing it, which is the point: an `nv_*` function that needs a dtype only
#' to decide something with (a category test, a `nan_rm` branch) must not force
#' the value to settle just by asking.
#'
#' Where the dtype becomes the operation's own -- what the other arguments are
#' converted to, or what the result is built `_like` -- the value has to commit
#' for real; that is a `promote` rule of [`as_anvl_arrays()`] (exact, because it
#' decides and converts in one step) or, failing that, [`as_anvl_array()`],
#' which converts at the default. Not this.
#'
#' @param x ([`arrayish`] | [`AbstractArray`])\cr
#'   The value to ask about.
#' @return ([`tengen::DataType`])
#' @seealso [as_anvl_arrays()], [RDataArray], [shape_abstract()]
#' @examplesIf pjrt::plugins_downloaded()
#' peek_dtype(1.5)
#' peek_dtype(1L)
#' peek_dtype(nv_array(1:3, dtype = "i8"))
#' @export
peek_dtype <- function(x) {
  aval <- to_abstract(x)
  if (is_rdata_array(aval)) aval$default_dtype else aval$dtype
}

# A box, with any R value in it committed to its default dtype. Anything that
# already has a dtype is returned unchanged.
commit_rdata_box <- function(x) {
  if (is_rdata_box(x)) commit_rdata(x) else x
}

# A fresh RData node for an R value written in the body of a traced function.
new_rdata_box <- function(desc, x) {
  shape <- if (is.null(dim(x))) integer() else as.integer(dim(x))
  GraphBox(GraphRData(RDataArray(x, shape = shape)), desc)
}


#' @export
format.GraphValue <- function(x, ...) {
  sprintf("GraphValue(%s)", format(x$aval))
}

#' @export
print.GraphValue <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
format.GraphLiteral <- function(x, ...) {
  # otherwise there might be conversion issues, so we directly use the pjrt printer
  # instead of converting via as_array(), which loses precision
  val <- if (is_anvl_array(x$aval$data)) {
    trimws(capture.output(print(x$aval$data))[2L])
  } else {
    as.character(x$aval$data)
  }
  sprintf("GraphLiteral(%s, %s, %s)", val, dtype2string(x$aval$dtype), shape2string(x$aval$shape))
}

#' @export
print.GraphLiteral <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @title Graph Node
#' @description
#' Virtual base class for nodes in an [`AnvlGraph`].
#' Is a [`GraphValue`], a [`GraphLiteral`], or -- only while tracing, and never
#' as part of a primitive call -- a [`GraphRData`].
#' Cannot be instantiated directly - use [`GraphValue()`] or [`GraphLiteral()`] instead.
#' @name GraphNode
NULL

#' @title Primitive Call
#' @description
#' Call of a primitive in an [`AnvlGraph`].
#' @param primitive (`AnvlPrimitive`)\cr
#'   The function.
#' @param inputs (`list(GraphValue)`)\cr
#'   The (array) inputs to the primitive.
#' @param params (`list(<any>)`)\cr
#'   The (static) parameters of the function call.
#' @param outputs (`list(GraphValue)`)\cr
#'   The (array) outputs of the primitive.
#' @return (`PrimitiveCall`)
#' @export
PrimitiveCall <- function(primitive, inputs, params, outputs) {
  if (inherits(primitive, "JitPrimitive")) {
    primitive <- attr(primitive, "primitive")
  }
  # hot-path constructor: no input validation
  structure(
    list(
      primitive = primitive,
      inputs = inputs,
      params = params,
      outputs = outputs
    ),
    class = "PrimitiveCall"
  )
}

#' @title Graph of Primitive Calls
#'
#' @description
#' Computational graph consisting exclusively of primitive calls.
#' This is a mutable class.
#'
#' @param calls (`list(PrimitiveCall)`)\cr
#'   The primitive calls that make up the graph.
#' @param in_tree (`NULL | Node`)\cr
#'   The tree of inputs. May contain leaves for both array inputs and static
#'   (non-array) arguments. Only the array leaves correspond to entries in
#'   `inputs`; use `is_static_flat` to distinguish them.
#' @param out_tree (`NULL | Node`)\cr
#'   The tree of outputs.
#' @param inputs (`list(GraphValue)`)\cr
#'   The inputs to the graph (array arguments only).
#' @param outputs (`list(GraphValue)`)\cr
#'   The outputs of the graph.
#' @param constants (`list(GraphValue)`)\cr
#'   The constants of the graph.
#' @param is_static_flat (`NULL | logical()`)\cr
#'   Boolean mask indicating which flat positions in `in_tree` are static (non-array) args.
#'   `NULL` when all args are array inputs.
#' @param static_args_flat (`NULL | list()`)\cr
#'   Flattened traced values for the static arguments indicated by `is_static_flat`.
#' @return (`AnvlGraph`)
# @export
AnvlGraph <- function(
  calls = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  constants = list(),
  is_static_flat = NULL,
  static_args_flat = NULL
) {
  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  env$calls <- calls
  env$in_tree <- in_tree
  env$out_tree <- out_tree
  env$inputs <- inputs
  env$outputs <- outputs
  env$constants <- constants
  env$is_static_flat <- is_static_flat
  env$static_args_flat <- static_args_flat

  structure(env, class = "AnvlGraph")
}

# The dtype each of the graph's inputs is supplied at, for the backends that
# have to upload it: one entry per input, `NA` for one the caller passes through
# unchanged (an array, which already has its dtype). Only an input built from
# bare R data names one -- it has no dtype of its own, so the program decided
# what it is uploaded as, and the aval remembers it (see [`RDataInput`]).
# `NULL` when no input came from R data, which is the overwhelmingly common
# case and lets the callers skip the whole step.
graph_input_dtypes <- function(graph) {
  is_rdata <- vapply(graph$inputs, function(gval) is_rdata_input(gval$aval), logical(1L))
  if (!any(is_rdata)) {
    return(NULL)
  }
  ifelse(is_rdata, vapply(graph$inputs, function(gval) as.character(gval$aval$dtype), character(1L)), NA_character_)
}

#' @title Graph Descriptor
#' @description
#' Descriptor of an [`AnvlGraph`]. This is a mutable class.
#' @param calls (`list(PrimitiveCall)`)\cr
#'   The primitive calls that make up the graph.
#' @param tensor_to_gval (`hashtab`)\cr
#'   Mapping: `AnvlArray` -> `GraphValue`
#' @param gval_to_box (`hashtab`)\cr
#'   Mapping: `GraphValue` -> `GraphBox`
#' @param constants (`list(GraphValue)`)\cr
#'   The constants of the graph.
#' @param in_tree (`NULL | Node`)\cr
#'   The tree of inputs. May contain leaves for both array inputs and static
#'   (non-array) arguments. Only the array leaves correspond to entries in
#'   `inputs`; use `is_static_flat` to distinguish them.
#' @param out_tree (`NULL | Node`)\cr
#'   The tree of outputs.
#' @param inputs (`list(GraphValue)`)\cr
#'   The inputs to the graph (array arguments only).
#' @param outputs (`list(GraphValue)`)\cr
#'   The outputs of the graph.
#' @param is_static_flat (`NULL | logical()`)\cr
#'   Boolean mask indicating which flat positions in `in_tree` are static (non-array) args.
#'   `NULL` when all args are array inputs.
#' @param static_args_flat (`NULL | list()`)\cr
#'   Flattened traced values for the static arguments indicated by `is_static_flat`.
#' @param devices (`character()`)\cr
#'   Device platforms encountered during tracing (e.g. `"cpu"`, `"cuda"`).
#'   Populated automatically as arrays are registered.
#' @return (`GraphDescriptor`)
#' @export
GraphDescriptor <- function(
  calls = list(),
  tensor_to_gval = NULL,
  gval_to_box = NULL,
  constants = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  is_static_flat = NULL,
  static_args_flat = NULL,
  devices = character()
) {
  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  # `calls` accumulates one entry per traced primitive. A fastqueue gives
  # amortised-O(1) append; growing an R list here (`env$calls[[n]] <- x` or
  # `c(env$calls, x)`) is copy-on-modify and would make tracing O(n^2).
  env$calls <- fastmap::fastqueue()
  if (length(calls)) {
    env$calls$madd(.list = calls)
  }
  env$data_to_gval <- tensor_to_gval %||% hashtab()
  env$gval_to_box <- gval_to_box %||% hashtab()
  env$constants <- constants
  env$in_tree <- in_tree
  env$out_tree <- out_tree
  env$inputs <- inputs
  env$outputs <- outputs
  env$is_static_flat <- is_static_flat
  env$static_args_flat <- static_args_flat
  env$devices <- devices
  # Calls that have to run before everything else, because they only depend on
  # the graph's inputs: the converts finalize_rdata_inputs() adds for an R
  # argument that one program used at more than one dtype.
  env$pre_calls <- list()

  structure(env, class = "GraphDescriptor")
}

#' @export
shape.GraphValue <- function(x, ...) {
  shape(x$aval)
}

#' @export
dtype.GraphValue <- function(x, ...) {
  dtype(x$aval)
}

#' @export
shape.GraphLiteral <- function(x, ...) {
  shape(x$aval)
}

#' @export
dtype.GraphLiteral <- function(x, ...) {
  x$aval$dtype
}


is_graph_descriptor <- function(x) {
  inherits(x, "GraphDescriptor")
}

descriptor_to_graph <- function(descriptor) {
  graph <- AnvlGraph(
    calls = c(descriptor$pre_calls, descriptor$calls$as_list()),
    in_tree = descriptor$in_tree,
    out_tree = descriptor$out_tree,
    inputs = descriptor$inputs,
    outputs = descriptor$outputs,
    constants = descriptor$constants,
    is_static_flat = descriptor$is_static_flat,
    static_args_flat = descriptor$static_args_flat
  )
  maybe_restore_previous_desc(descriptor)
  graph
}

# Now the graph-building

#' @title Graph Box
#' @description
#' An [`AnvlBox`] subclass that wraps a [`GraphNode`] during graph construction (tracing).
#' When a function is traced via [`trace_fn()`], each intermediate array
#' value is represented as a `GraphBox`.
#' It also contains an associated [`GraphDescriptor`] in which the node "lives".
#'
#' @section Extractors:
#' - [`dtype()`][tengen::dtype]
#' - [`shape()`][tengen::shape]
#' - [`naxes()`][tengen::naxes]
#'
#' @param gnode ([`GraphNode`])\cr
#'   The graph node -- either a [`GraphValue`] or a [`GraphLiteral`].
#' @param desc ([`GraphDescriptor`])\cr
#'   The descriptor of the graph being built.
#' @return (`GraphBox`)
#'
#' @seealso [AnvlBox], [trace_fn()], [jit()]
#' @export
GraphBox <- function(gnode, desc) {
  # hot-path constructor: no input validation
  structure(
    list(gnode = gnode, desc = desc),
    class = c("GraphBox", "AnvlBox")
  )
}

#' @export
shape.GraphBox <- function(x, ...) {
  shape(x$gnode)
}

#' @export
dtype.GraphBox <- function(x, ...) {
  dtype(x$gnode)
}

#' @export
backend.GraphBox <- function(x, ...) {
  # Tracing is backend-agnostic
  "plain"
}

#' @export
device.GraphBox <- function(x, ...) {
  cli_abort(c(
    "{.fn device} is not defined for a {.cls GraphBox}.",
    i = "During tracing there is no concrete device; jit handles device placement at the input/output boundary and for constants.",
    i = "If you need a constant on the same device as an arrayish input, use {.fn nv_fill_like} / {.fn nv_array_like} / {.fn nv_iota_like}, which pick the device up from the tracing context for you."
  ))
}

#' @export
print.GraphBox <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
format.GraphBox <- function(x, ...) {
  sprintf("GraphBox(%s)", format(x$gnode))
}

# Box an arrayish value for `desc`.
#
# `materialize = FALSE` keeps an R value as an [`RDataArray`] -- with no dtype,
# to be built at the one its use site needs. It is for the API layer, which
# canonicalizes its inputs (`as_anvl_array()`) long before it knows what they
# are combined with. Everything that needs a typed value -- every primitive
# call, and the outputs of a trace -- takes the default and commits it.
maybe_box_arrayish <- function(x, desc = .current_descriptor(), materialize = TRUE) {
  if (is_graph_box(x)) {
    if (is_graph_rdata(x$gnode)) {
      # An R value belongs to the graph it was written in, so one reaching
      # another graph has to commit before it can be captured there.
      if (materialize || !identical(x$desc, desc)) {
        x <- commit_rdata(x)
      } else {
        return(x)
      }
    }
    if (identical(x$desc, desc)) {
      return(x)
    }
    return(get_box_or_register_const(desc, x$gnode))
  }
  if (!materialize && (is_valid_r_lit(x) || is_valid_r_array(x))) {
    return(new_rdata_box(desc, x))
  }
  if (is_valid_r_array(x)) {
    # Materialize R arrays as plain-backend AnvlArrays so they can be
    # registered as named constants in the current graph.
    x <- nv_array(x)
  }
  if (is_anvl_array(x) || is_valid_r_lit(x)) {
    return(get_box_or_register_const(desc, x))
  }
  cli_abort("Expected arrayish value, but got {.cls {class(x)[1]}}")
}

# Called only by trace_fn() to wire up each flat arg as an input of `desc`.
# Behavior is fully determined by `mode`:
# - "toplevel": jit's outermost trace. No parent descriptor exists. Arrayish
#   args become fresh input gvals; non-arrayish args pass through as static
#   parameters.
# - "subgraph": traced body of a higher-order primitive (prim_if/prim_while/...).
#   The caller promises all args are arrayish, so R lits/arrays are promoted
#   to AnvlArrays. Each arrayish arg becomes a fresh AbstractArray-typed
#   gval -- a clean parameter slot for the subgraph. Non-arrayish args are
#   an error.
# - "inline": traced graph that will be later be inlined into the paren
#   (gradient/value_and_gradient). Inputs that are not already boxed
#   are registered in the parent graph and then the inputs alias them
#   which simplified subsequent inlining
maybe_box_input <- function(x, desc, mode) {
  if (mode == "subgraph") {
    # e.g.: prim_while(list(i = 1), ...)
    # we know which inputs are dynamic/static -> convert
    if (is_valid_r_lit(x)) {
      x <- nv_scalar(x)
    } else if (is_valid_r_array(x)) {
      x <- nv_array(x)
    }
    # e.g.: prim_while(list(i = nv_scalar(1)), ...)
    if (is_anvl_array(x)) {
      if (backend(x) != "plain") {
        desc$devices <- c(desc$devices, device(x))
      }
      gval <- GraphValue(aval = to_abstract(x, pure = TRUE))
      return(register_input(desc, gval))
    }
    # e.g.: \(x) prim_while(list(i = x), ...)
    if (is_graph_box(x)) {
      # A subgraph parameter needs a dtype, and the subgraph is traced before
      # its operands meet anything, so an R value commits here.
      if (is_graph_rdata(x$gnode)) {
        x <- commit_rdata(x)
      }
      gval <- GraphValue(aval = abstract_aval(x$gnode$aval))
      return(register_input(desc, gval))
    }
    # is used internally by prim_scatter() to trace `update_computation()` with avals
    if (is_abstract_array(x)) {
      gval <- GraphValue(aval = x)
      return(register_input(desc, gval))
    }
    cli_abort("In subgraph mode, all args must be arrayish; got {.cls {class(x)[1]}}")
  }

  if (mode == "inline") {
    # gradient(f)(nv_scalar(1))
    if (is_anvl_array(x)) {
      if (backend(x) != "plain") {
        desc$devices <- c(desc$devices, device(x))
      }
      parent_desc <- maybe_previous_descriptor()
      parent_box <- get_box_or_register_const(parent_desc, x)
      return(register_input(desc, parent_box$gnode))
    }
    # \(x) gradient(f)(x)
    if (is_graph_box(x)) {
      if (is_graph_rdata(x$gnode)) {
        x <- commit_rdata(x)
      }
      return(register_input(desc, x$gnode))
    }
    # don't convert R values because they might be static.
    # We don't know because gradient() does not annotate static
    return(x)
  }

  # mode == "toplevel"
  if (is_anvl_array(x)) {
    if (backend(x) != "plain") {
      desc$devices <- c(desc$devices, device(x))
    }
    gval <- GraphValue(aval = to_abstract(x, pure = TRUE))
    return(register_input(desc, gval))
  }
  if (is_rdata_array(x)) {
    # Bare R data passed to a jitted function. It takes an input slot like any
    # other argument -- the call has to supply the value -- but which dtype
    # that input has is only known once the body has used it, so the slot is
    # filled in by finalize_rdata_inputs().
    return(register_rdata_input(desc, x))
  }
  if (is_graph_box(x)) {
    if (is_graph_rdata(x$gnode)) {
      x <- commit_rdata(x)
    }
    return(register_input(desc, x$gnode))
  }
  if (is_abstract_array(x)) {
    gval <- GraphValue(aval = x)
    return(register_input(desc, gval))
  }
  x
}

# Strip data from a (possibly concrete) array aval, returning a pure
# AbstractArray with the same dtype and shape.
abstract_aval <- function(aval) {
  if (is_concrete_tensor(aval)) {
    AbstractArray(dtype = aval$dtype, shape = aval$shape)
  } else {
    aval
  }
}

register_input <- function(desc, x) {
  if (!is_graph_descriptor(desc)) {
    cli_abort("Internal error: trying to register an input in a non-graph descriptor")
  }
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid input")
  }
  desc$inputs <- c(desc$inputs, list(x))
  box <- GraphBox(x, desc)
  desc$gval_to_box[[x]] <- box
  box
}

# Reserve `desc`'s next input slot for an R argument. The slot holds the
# GraphRData node itself until finalize_rdata_inputs() replaces it with the
# value the body materialized, which keeps the input order the same as the
# argument order (the caller supplies its inputs in that order).
register_rdata_input <- function(desc, aval) {
  node <- GraphRData(aval)
  desc$inputs <- c(desc$inputs, list(node))
  GraphBox(node, desc)
}

# Settle every R argument of a finished trace: which dtype its input is
# supplied at, and the converts for any further dtype the body used it at. The
# resolved dtype is written onto the input's own aval, as an [`RDataInput`], so
# the finished graph carries it where the input is rather than beside it.
#
# The upload dtype is the highest precision the body asked for, staying in the
# R value's own category where it can -- an R integer used as both `i64` and
# `f32` is uploaded as `i64`, so the `i64` use site is exact and the `f32` one
# rounds once, as it would have from the R value itself. A value the body never
# used commits to its default, so the input is still there for the caller to
# supply.
finalize_rdata_inputs <- function(desc) {
  inputs <- desc$inputs
  is_rdata <- vapply(inputs, is_graph_rdata, logical(1L))
  if (!any(is_rdata)) {
    return(invisible(NULL))
  }
  pre_calls <- list()
  for (i in which(is_rdata)) {
    node <- inputs[[i]]
    aval <- node$aval
    # Only the dtypes the value was *built* at can be uploaded; the memo also
    # holds the results of converting out of its category, which the program
    # computes from one of these.
    requested <- Filter(
      function(dt) rdata_builds_directly(aval$r_type, as_dtype(dt)),
      names(node$mat)
    )
    resolved <- resolve_upload_dtype(aval, requested)
    main <- node$mat[[resolved]] %||%
      GraphBox(GraphValue(AbstractArray(resolved, aval$shape)), desc)
    main$gnode$aval <- RDataInput(resolved, aval$shape, aval$r_type)
    inputs[[i]] <- main$gnode
    for (other in setdiff(requested, resolved)) {
      pre_calls[[length(pre_calls) + 1L]] <- PrimitiveCall(
        primitive = prim_convert,
        inputs = list(main$gnode),
        params = list(dtype = as_dtype(other)),
        outputs = list(node$mat[[other]]$gnode)
      )
    }
  }
  desc$inputs <- inputs
  desc$pre_calls <- c(desc$pre_calls, pre_calls)
  invisible(NULL)
}

# The single dtype an R argument is uploaded at, given every dtype the trace
# built it at. Those are all in the value's own category (materialize_rdata()
# converts out of it inside the program instead), so the widest of them holds
# every use site's value: a narrower site converts down from an exact upload,
# which rounds exactly once.
resolve_upload_dtype <- function(aval, requested) {
  if (!length(requested)) {
    return(as.character(aval$default_dtype))
  }
  dtypes <- lapply(requested, as_dtype)
  widths <- vapply(dtypes, dtype_width, integer(1L))
  # `order()` breaks a width tie by the dtype name, so the choice does not
  # depend on the order the trace happened to ask in.
  as.character(dtypes[[order(-widths, vapply(dtypes, as.character, character(1L)))[[1L]]]])
}

register_gvals <- function(desc, gvals) {
  lapply(gvals, register_gval, desc = desc)
}

register_gval <- function(desc, x) {
  # hot path (one call per traced output): no input validation
  box <- desc$gval_to_box[[x]]
  if (!is.null(box)) {
    return(box)
  }
  box <- GraphBox(x, desc)
  desc$gval_to_box[[x]] <- box
  box
}

# Returns a Box
get_box_or_register_const <- function(desc, x) {
  if (is_anvl_array(x)) {
    if (backend(x) != "plain") {
      desc$devices <- c(desc$devices, device(x))
    }
    gval <- desc$data_to_gval[[x]]
    if (!is.null(gval)) {
      return(desc$gval_to_box[[gval]])
    }
    gval <- GraphValue(aval = ConcreteArray(x))
    desc$data_to_gval[[x]] <- gval
    desc$constants <- c(desc$constants, list(gval))
    box <- GraphBox(gval, desc)
    desc$gval_to_box[[gval]] <- box
    return(box)
  }
  if (is_valid_r_lit(x)) {
    gval <- GraphLiteral(LiteralArray(x, shape = integer()))
    box <- desc$gval_to_box[[gval]] <- GraphBox(gval, desc)
    return(box)
  }
  if (is_graph_literal(x)) {
    box <- desc$gval_to_box[[x]] <- GraphBox(x, desc)
    return(box)
  }
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid constant")
  }
  # gval$aval can either be a
  # * ConcreteArray: AnvlArray that is captured from the parent environment
  # * AbstractArray: Output of a computation in a parent graph
  # In either case, we first check whether the value is already registered in the current graph
  # and if so, return it:
  box <- desc$gval_to_box[[x]]
  if (!is.null(box)) {
    return(box)
  }

  # Now, we create the new box and register it, so if we see it again, we can return it immediately.
  new_box <- GraphBox(x, desc)

  if (is_concrete_tensor(x$aval)) {
    desc$data_to_gval[[x$aval$data]] <- x
  }
  desc$gval_to_box[[x]] <- new_box
  desc$constants <- c(desc$constants, list(x))
  return(new_box)
}

register_inputs <- function(desc, inputs) {
  for (input in inputs) {
    register_input(desc, input)
  }
}

register_consts <- function(desc, consts) {
  for (const in consts) {
    get_box_or_register_const(desc, const)
  }
}

match_args_to_formals <- function(f, args) {
  g <- function() {
    as.list(match.call()[-1L])
  }
  formals(g) <- formals(f)
  do.call(g, args)
}

#' @title Trace an R function into a Graph
#' @description
#' Executes `f` with abstract array arguments and records every primitive operation into
#' an [`AnvlGraph`].
#'
#' The resulting graph can be lowered to StableHLO (via [`stablehlo()`]) or transformed
#' (e.g. via [`transform_gradient()`]).
#'
#' @param f (`function`)\cr
#'   The function to trace. Must not be a `JitFunction` (i.e. already jitted).
#' @param args (`list` of ([`AnvlArray`] | [`AbstractArray`]))\cr
#'   The (unflattened) arguments to the function. Mutually exclusive with the
#'   `args_flat`/`in_tree` pair.
#' @param desc (`NULL` | `GraphDescriptor`)\cr
#'   Optional descriptor. When `NULL` (default), a new descriptor is created.
#' @param mode (`character(1)`)\cr
#'   How to handle the inputs.
#'   Options are:
#'   - `"toplevel"`: Used for jit(). Default.
#'   - `"subgraph"`: Use for tracing subgraphs in higher-order primitives like [`prim_while()`].
#'   - `"inline"`: Use for transformations like jit, where the graph is later inlined
#'     into the parent graph.
#' @param args_flat (`list`)\cr
#'   Flattened arguments. Must be accompanied by `in_tree`.
#' @param in_tree (`Node`)\cr
#'   Tree structure describing how `args_flat` maps back to `f`'s arguments.
#' @template param_optimize
#' @return An [`AnvlGraph`] containing the traced operations.
#' @seealso [`stablehlo()`] to lower the graph, [`jit()`] for end-to-end
#'   compilation.
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' graph <- trace_fn(function(x, y) x + y,
#'   args = list(x = nv_array(1, dtype = "f32"), y = nv_array(2, dtype = "f32")))
#' graph
trace_fn <- function(
  f,
  args = NULL,
  desc = NULL,
  mode = NULL,
  args_flat = NULL,
  in_tree = NULL,
  optimize = FALSE
) {
  if (is.null(mode) && !currently_tracing()) {
    mode <- "toplevel"
  }
  mode <- assert_choice(mode, c("toplevel", "subgraph", "inline"))
  if (is.null(args)) {
    if (is.null(args_flat) || is.null(in_tree)) {
      cli_abort("args or args_flat and in_tree must be provided")
    }
  } else {
    if (!is.null(args_flat) || !is.null(in_tree)) {
      cli_abort("args and args_flat and in_tree must not be provided together")
    }
    # Match args with parameters of f before flattening
    args <- match_args_to_formals(f, args)
    in_tree <- build_tree(args)
    args_flat <- flatten(args)
  }
  f_flat <- pjrt::flatten_fun(f, in_tree = in_tree)
  if (is.null(desc)) {
    desc <- local_descriptor(in_tree = in_tree)
  } else {
    desc$in_tree <- in_tree
  }

  parent_desc <- maybe_previous_descriptor()
  if (mode == "toplevel" && !is.null(parent_desc)) {
    cli_abort('Internal error: trace_fn(mode = "toplevel") must not have a parent descriptor')
  }
  if (mode != "toplevel" && is.null(parent_desc)) {
    cli_abort('Internal error: trace_fn(mode = "{mode}") requires a parent descriptor')
  }

  # box arrays and add them as inputs to the current graph
  inputs_flat <- lapply(args_flat, maybe_box_input, desc = desc, mode = mode)
  # Track which flat args are static (non-array) values vs. graph inputs
  desc$is_static_flat <- vapply(inputs_flat, Negate(is_graph_box), logical(1L))
  if (mode == "toplevel") {
    globals[["INFER_PRIMITIVE"]] <- NULL
    output <- tryCatch(
      do.call(f_flat, inputs_flat),
      error = function(e) {
        prim <- globals[["INFER_PRIMITIVE"]]
        globals[["INFER_PRIMITIVE"]] <- NULL
        if (!is.null(prim)) {
          e$call <- print_call_repr(prim)
          # only stablehlo errors carry 0-based indices to convert
          if (inherits(e, "ErrorStablehlo")) {
            e <- stablehlo::to_one_based(e)
          }
          e <- to_user_terminology(e)
        }
        rlang::cnd_signal(e)
      }
    )
  } else {
    output <- do.call(f_flat, inputs_flat)
  }

  out_tree <- output[[1L]]
  # function() x; -> output can be an closed-over constant
  outputs_flat <- lapply(output[[2L]], maybe_box_arrayish)

  desc$out_tree <- out_tree
  desc$outputs <- lapply(outputs_flat, \(x) x$gnode)
  if (mode == "toplevel") {
    finalize_rdata_inputs(desc)
  }
  if (!is.null(desc$is_static_flat) && isTRUE(any(desc$is_static_flat))) {
    desc$static_args_flat <- args_flat[desc$is_static_flat]
  } else {
    desc$static_args_flat <- NULL
  }

  graph <- descriptor_to_graph(desc)
  optimize_graph(graph, optimize)
}

is_graph_value <- function(x) {
  inherits(x, "GraphValue")
}

maybe_restore_previous_desc <- function(desc = NULL) {
  if (!is.null(desc) && (!identical(desc, globals[["CURRENT_DESCRIPTOR"]]))) {
    # graph has already been returned
    return()
  }

  stash_size <- length(globals[["DESCRIPTOR_STASH"]])
  if (stash_size) {
    globals[["CURRENT_DESCRIPTOR"]] <- globals[["DESCRIPTOR_STASH"]][[stash_size]]
    globals[["DESCRIPTOR_STASH"]] <- globals[["DESCRIPTOR_STASH"]][-stash_size]
  } else {
    globals[["CURRENT_DESCRIPTOR"]] <- NULL
  }
}

#' @title Get the current graph
#' @description
#' Get the current graph being built (via [`local_descriptor`]).
#' @param silent (`logical(1)`)\cr
#'   Whether to return `NULL` if no graph is currently being built (as opposed to aborting).
#' @return A [`GraphDescriptor`] object.
#' @export
.current_descriptor <- function(silent = FALSE) {
  maybe_desc <- globals[["CURRENT_DESCRIPTOR"]]
  if (silent) {
    return(maybe_desc)
  }
  maybe_desc %||%
    cli_abort("No graph is currently being built. Did you forget to use `jit()`?")
}

currently_tracing <- function() {
  # read the global directly: this runs on every jitted call (hot path)
  !is.null(globals[["CURRENT_DESCRIPTOR"]])
}

maybe_previous_descriptor <- function() {
  stash <- globals[["DESCRIPTOR_STASH"]]
  n <- length(stash)
  if (!n) {
    return(NULL)
  }
  stash[[n]]
}

#' @title Create a graph
#' @description
#' Creates a new [`GraphDescriptor`] which is afterwards accessible via [`.current_descriptor()`].
#' The graph is automatically removed when exiting the current scope.
#' After the graph is either cleaned up automatically (by exiting the scope)
#' or finalized, the previously built graph is restored,
#' i.e., accessible via [`.current_descriptor()`].
#'
#' @param envir (`environment`)\cr
#'   Environment where exit handler will be registered for cleaning up the
#'   [`GraphDescriptor`] if it was not returned yet.
#' @param ... (`any`)\cr
#'   Additional arguments to pass to the [`GraphDescriptor`] constructor.
#' @return A [`GraphDescriptor`] object.
#' @export
local_descriptor <- function(..., envir = parent.frame()) {
  if (identical(envir, globalenv())) {
    # lingering global descriptors interfere with graph tracing
    cli_abort("Don't run local_descriptor in the global environment")
  }

  desc <- GraphDescriptor(...)
  if (!is.null(globals[["CURRENT_DESCRIPTOR"]])) {
    globals[["DESCRIPTOR_STASH"]] <- c(
      globals[["DESCRIPTOR_STASH"]],
      list(globals[["CURRENT_DESCRIPTOR"]])
    )
  }
  globals[["CURRENT_DESCRIPTOR"]] <- desc

  withr::defer(
    envir = envir,
    {
      maybe_restore_previous_desc(desc)
    },
    priority = "first"
  )
  return(desc)
}

is_graph <- function(x) {
  inherits(x, "AnvlGraph")
}
is_graph_box <- function(x) {
  inherits(x, "GraphBox")
}

#' @title Add a Primitive Call to a Graph Descriptor
#' @description
#' Add a primitive call to a graph descriptor. Inside a primitive body created
#' with [`new_primitive()`], pass the lexically-bound `self` as the primitive
#' argument.
#' @param primitive ([`AnvlPrimitive`] | `JitPrimitive`)\cr
#'   The primitive the call is for. A `JitPrimitive` is accepted and unwrapped
#'   to its underlying `AnvlPrimitive` metadata.
#' @param args (`list` of [`GraphNode`])\cr
#'   The arguments to the primitive.
#' @param params (`list`)\cr
#'   The parameters to the primitive.
#' @param infer_fn (`function`)\cr
#'   The inference function to use.
#'   Must output a list of [`AbstractArray`]s.
#' @param desc ([`GraphDescriptor`] | `NULL`)\cr
#'   The graph descriptor to add the primitive call to.
#'   Uses the [current descriptor][.current_descriptor] if `NULL`.
#' @return (`list` of [`GraphBox`])
#' @export
graph_desc_add <- function(primitive, args, params = list(), infer_fn, desc = NULL) {
  desc <- desc %||% .current_descriptor(silent = TRUE)
  if (inherits(primitive, "JitPrimitive")) {
    primitive <- attr(primitive, "primitive")
  }

  # Box each input and pull out its gnode + aval in one pass (`gnodes_in`
  # unnamed for the PrimitiveCall; `avals_in` keeps arg names for infer_fn).
  n_in <- length(args)
  gnodes_in <- vector("list", n_in)
  avals_in <- vector("list", n_in)
  for (i in seq_len(n_in)) {
    gnode <- maybe_box_arrayish(args[[i]], desc)$gnode
    gnodes_in[[i]] <- gnode
    avals_in[[i]] <- gnode$aval
  }
  names(avals_in) <- names(args)
  globals[["INFER_PRIMITIVE"]] <- primitive
  ats_out <- do.call(infer_fn, c(avals_in, params))
  globals[["INFER_PRIMITIVE"]] <- NULL
  gvals_out <- lapply(ats_out, GraphValue)
  call <- PrimitiveCall(primitive, gnodes_in, params, gvals_out)
  desc$calls$add(call)
  lapply(gvals_out, register_gval, desc = desc)
}

print_call_repr <- function(prim) {
  rlang::exec(call, paste0("prim_", prim$name))
}

# Restate a type-inference error in anvl's own vocabulary: stablehlo speaks of
# "tensors" and names its primary argument `operand`, anvl speaks of arrays and
# names it `x`. Both message paths have to be covered: `ErrorStablehlo`
# subclasses build their message lazily in `conditionMessage()` methods, while
# `cli_abort()` conditions store an already formatted message in the `message`
# and `body` fields, which `rlang::cnd_message()` reads without dispatching on
# `conditionMessage()`.
to_user_terminology <- function(x) {
  if (!inherits(x, "condition")) {
    return(x)
  }
  if (is.character(x$message)) {
    x$message <- user_terminology(x$message)
  }
  if (is.character(x$body)) {
    x$body <- user_terminology(x$body)
  }
  class(x) <- c("AnvlErrorTerminology", class(x))
  x
}

#' @export
conditionMessage.AnvlErrorTerminology <- function(c, ...) {
  user_terminology(NextMethod())
}

# The substitutions are anchored on word boundaries. `_` is a word character,
# so identifiers such as `TensorType`, `hlo_tensor()` or `operand_batching_dims`
# contain no boundary around the word and are left alone.
user_terminology <- function(x) {
  x <- gsub("\\btensor(s?)\\b", "array\\1", x)
  x <- gsub("\\bTensor(s?)\\b", "Array\\1", x)
  gsub("\\boperand\\b", "x", x)
}


inline_graph_into_desc <- function(desc, graph) {
  # By contract, `graph` was produced by `trace_fn(..., mode = "inline")` so
  # every input gval was registered in `desc` at trace time and is already
  # known to the parent. Only sub-graph constants (closed-over values
  # registered via `maybe_box_arrayish`) still need to be propagated up.
  register_consts(desc, graph$constants)

  if (length(graph$calls)) {
    desc$calls$madd(.list = graph$calls)
  }

  gvals_out_flat <- graph$outputs
  boxes_out_flat <- lapply(gvals_out_flat, GraphBox, desc)
  unflatten(graph$out_tree, boxes_out_flat)
}
