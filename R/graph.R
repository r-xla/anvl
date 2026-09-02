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

#' @title R Data Class
#' @description
#' The [`AbstractArray`] of a non-static R value that was passed to a jit-compiled
#' function.
#' It is special, because it does not have a data type, i.e., calling `dtype()` on it
#' results in an error.
#' @details
#' It only exists *during* tracing, never in a finalized graph.
#' @section Extractors:
#' Just like [`AbstractArray`], with the exception that `dtype()` errs.
#'
#' @param shape ([`stablehlo::Shape`] | `integer()`)\cr
#'   The shape of the value: `()` for a length-1 vector, its `dim()` for an
#'   R array.
#' @param r_type (`character(1)`)\cr
#'   The R storage type: `"double"`, `"integer"` or `"logical"`.
#'
#' @return (`RData`)
#' @seealso [AbstractArray], [AnvlGraph]
#' @examplesIf pjrt::plugins_downloaded()
#' RData(c(2, 3), "double")
#' # is equivalent to
#' nv_aval("double", c(2, 3))
#' # Below, the `RData` input is materialized in 32 and 64-bit precisions, so the input
#' # dtype becomes f64.
#' # By NOT converting RData to their default data type we prevent loss of precision
#' # (double -> f32 -> f64 roundrips)
#' graph <- trace_fn(function(x) {
#'     print(x)
#'     list(x + nv_scalar(1, "f64"), x + nv_scalar(1, "f32"))
#'   }, list(x = nv_aval("double", c()))
#' )
#' print(graph)
#' # The actual inputs to the compiled program
#' graph$inputs
#' # The data types of the R values; AnvlArrays get NA here
#' graph$rdata_types
#' @export
RData <- function(shape, r_type) {
  shape <- as_shape(shape)
  r_type <- match.arg(r_type, c("double", "integer", "logical"))
  structure(
    list(
      r_type = r_type,
      default_dtype = default_dtype_r(r_type),
      shape = shape
    ),
    class = c("RData", "AbstractArray")
  )
}

is_rdata <- function(x) {
  inherits(x, "RData")
}

#' @method dtype RData
#' @export
dtype.RData <- function(x, ...) {
  abort_no_dtype(x$default_dtype)
}


#' @method shape numeric
#' @export
shape.numeric <- function(x, ...) {
  if (!is.null(dim(x))) {
    return(as.integer(dim(x)))
  }
  if (length(x) != 1L) {
    cli_abort(c(
      "{.fn shape} is undefined for a length-{length(x)} R vector.",
      i = "Only a length-1 R value and an {.fn array} are arrayish; use {.fn nv_array} to make one an array."
    ))
  }
  integer()
}

#' @method shape logical
#' @export
shape.logical <- shape.numeric

#' @method dtype numeric
#' @export
dtype.numeric <- function(x, ...) {
  abort_no_dtype(default_dtype(x))
}

#' @method dtype logical
#' @export
dtype.logical <- function(x, ...) {
  abort_no_dtype(default_dtype(x))
}

abort_no_dtype <- function(default_dtype) {
  cli_abort(
    c(
      "An R value has no data type of its own until it is used.",
      i = "{.fn dtype} is undefined here for the same reason {.code dtype(1.5)} is: the value only takes a data type when it meets a typed array, or when it commits to the default ({.val {as.character(default_dtype)}}).", # nolint
      i = "Give it one explicitly with {.fn nv_convert}."
    ),
    call = NULL
  )
}

#' @export
format.RData <- function(x, ...) {
  sprintf("RData(%s, %s)", x$r_type, shape2string(x$shape))
}

#' @export
print.RData <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @export
repr.RData <- function(x, ...) {
  sprintf("%s[%s]", x$r_type, repr(x$shape))
}

is_rdata_box <- function(x) {
  inherits(x, "GraphBox") && is_rdata(x$gnode$aval)
}

# The values the trace built an open R argument at, keyed by dtype name. Lives
# on the descriptor that registered the input rather than on the value, so a
# GraphValue stays a value.
rdata_mat <- function(desc, gval) {
  desc$rdata_mat[[gval]] %||% list()
}

rdata_mat_set <- function(desc, gval, key, box) {
  mat <- rdata_mat(desc, gval)
  mat[[key]] <- box
  desc$rdata_mat[[gval]] <- mat
  invisible(box)
}

rdata_natural_dtype <- function(r_type) {
  switch(r_type, double = as_dtype("f64"), integer = as_dtype("i32"), logical = as_dtype("bool"))
}

# The dtype an R value is built at on its way to `dtype`, for a `dtype` it
# cannot be built at directly. Building it where it is exact is what keeps the
# conversion the program's rather than R's -- but where that is wider than the
# dtype the value would commit to on its own, the program acquires one nothing
# asked for. An R double reaches `i32` through `f64`, so a program with no `f64`
# in it gets one, and a backend without `f64` (Metal) cannot run it at all.
#
# That is worth saying out loud rather than refusing: the alternative rounds the
# value through its default before converting, which is a different answer
# (`2^25 + 1` reaches `i32` as 33554432 rather than 33554433), and the caller may
# well want the exact one. `double` is the only R type this can happen for --
# an integer and a logical are exact at their own defaults.

# This is one of THE core functios of the RData mechanism, as it specifies how
# we obtain RData inputs at the requested data type.
# When an RData input is only requested at it's category, we use the dtype within that category
# that holds enough precision so everybody gets the value at its requested dtype without losing precision.
# To understand this, consider a program that receives an R double and compare two programs:
# 1. a single primitive call uses the R double and requires it in f32 -> input becomes f32
# 2. one primitive requires f32, the other f64. -> input becomes f64
#    this only works, because the f32 obtained from the f32 is the same as we obtain from the f64.
# Where we do cross-category concversions, a double input must e.g. always be materialized at
# f64, otherwise the result of an i32 request might depend on the float requests of that double,
# which MUST NOT HAPPEN. This is also why prim_convert(1, "i32") errs:
# We need to introduce an f64 into the program although nobody ever requested one, but otherwise
# would be even worse.
rdata_staging_dtype <- function(r_type, dtype) {
  staged <- rdata_natural_dtype(r_type)
  if (staged != default_dtype_r(r_type)) {
    cli_warn(
      c(
        "Converting an R {r_type} to {.val {as.character(dtype)}} brings {.val {as.character(staged)}} into the program.", # nolint
        x = "An R {r_type} cannot be built at {.val {as.character(dtype)}} directly, so it is built at {.val {as.character(staged)}} and the program converts -- and nothing else here asked for {.val {as.character(staged)}}.", # nolint
        i = "To keep it out, convert in its own category first: {.code nv_convert(nv_convert(x, {.str {as.character(default_dtype_r(r_type))}}), {.str {as.character(dtype)}})}. The result differs for values its data type cannot hold exactly." # nolint
      ),
      class = "anvl_staging_widens_warning"
    )
  }
  staged
}

rdata_in_category <- function(r_type, dtype) {
  dtype_category(dtype) == dtype_category(default_dtype_r(r_type))
}

# TODO: bit64 support
rdata_builds_directly <- function(r_type, dtype) {
  rdata_in_category(r_type, dtype) &&
    (r_type != "integer" || (is_dtype_int(dtype) && dtype_width(dtype) >= 32L))
}

# Bring an R value of storage type `r_type` into the program at `dtype`. `build`
# makes it at a data type it can be built at faithfully; a target it cannot is
# reached by building at the natural one and letting the *program* convert the
# rest of the way, so narrowing follows XLA's semantics rather than R's.
#
# The three ways an R value enters a program -- a literal in a traced body, the
# input an open argument is supplied at, an array built eagerly -- differ only
# in `build`, and this is what they share.
build_r_staged <- function(r_type, dtype, build) {
  if (rdata_builds_directly(r_type, dtype)) {
    return(build(dtype))
  }
  prim_convert(build(rdata_staging_dtype(r_type, dtype)), dtype = dtype)
}

# Build `box`'s R argument into the graph at `dtype`, and return the GraphBox
# for it. Memoized per dtype on the node: an argument used twice at one dtype
# takes one input.
materialize_rdata <- function(box, dtype) {
  hit <- rdata_mat_hit(box, dtype)
  if (!is.null(hit)) {
    return(hit)
  }
  out <- build_r_staged(box$gnode$aval$r_type, dtype, function(dt) rdata_input_at(box, dt))
  rdata_mat_set(box$desc, box$gnode, as.character(dtype), out)
}

# The input `box`'s R argument is supplied at, for a dtype it can be uploaded at
# directly. Its value is unknown here (the compiled program must not depend on
# it), so it becomes an input of this dtype and the call uploads the R data at
# it; `finalize_rdata_inputs()` puts it in the input list. Memoized like the
# converts above it, so a staged build and a later request for the same natural
# dtype share one input.
rdata_input_at <- function(box, dtype) {
  hit <- rdata_mat_hit(box, dtype)
  if (!is.null(hit)) {
    return(hit)
  }
  aval <- box$gnode$aval
  rdata_mat_set(
    box$desc,
    box$gnode,
    as.character(dtype),
    register_gval(box$desc, GraphValue(AbstractArray(dtype = dtype, shape = aval$shape)))
  )
}

# A memoized box is only reusable where it can actually be reached. One built in
# the node's own descriptor can: a sub-graph captures it like any other outer
# value. A `prim_convert` is recorded in whatever descriptor was current,
# though, so a sibling sub-graph -- `prim_if()`'s other branch, `prim_while()`'s
# body after its condition -- has to build its own rather than reference a value
# only the first one computes.
rdata_mat_hit <- function(box, dtype) {
  hit <- rdata_mat(box$desc, box$gnode)[[as.character(dtype)]]
  if (is.null(hit)) {
    return(NULL)
  }
  reachable <- identical(hit$desc, box$desc) || identical(hit$desc, .current_descriptor(silent = TRUE))
  if (reachable) hit else NULL
}

# Build the bare R value `x` -- an actual length-1 vector or array -- into
# `desc` at `dtype` as a constant, and return its GraphBox. Tracing only. The
# value is built from the R data itself, so it arrives with every digit it had,
# which is what keeps `x_f64 / sqrt(2)` exact.
build_r_at <- function(x, dtype, desc = .current_descriptor()) {
  force(desc)
  if (!is_valid_r_lit(x) && !is_valid_r_array(x)) {
    # An `NA` reaches here: it is a length-1 numeric, but there is no dtype it
    # obviously belongs at. Refused the same way it is anywhere else a value
    # enters a graph.
    cli_abort("Expected arrayish value, but got {.cls {class(x)[1]}}")
  }
  build_r_staged(typeof(x), as_dtype(dtype), function(dt) r_const_at(x, dt, desc))
}

# The graph's constant for the R value `x` at `dtype`: an inlined literal for a
# scalar, a registered constant for an array. Deduplicated per descriptor, so
# the same value written twice is built once.
r_const_at <- function(x, dtype, desc) {
  if (is.null(dim(x))) {
    return(get_box_or_register_const(
      desc,
      GraphLiteral(LiteralArray(x, shape = shape(x), dtype = dtype))
    ))
  }
  get_box_or_register_const(desc, nv_array(x, dtype = dtype))
}

#' @title Peek at a Data Type
#' @description
#' The data type `x` would use if was conveted to an `AnvlArray`.
#' Relevant for R objects and their [`RData`] trace-time analogon.
#'
#' @param x ([`arrayish`] | [`AbstractArray`])\cr
#'   The value to ask about.
#' @return ([`tengen::DataType`])
#' @seealso [as_anvl_arrays()], [RData], [shape()][tengen::shape]
#' @examplesIf pjrt::plugins_downloaded()
#' peek_dtype(1.5)
#' peek_dtype(1L)
#' peek_dtype(nv_array(1:3, dtype = "i8"))
#' @export
peek_dtype <- function(x) {
  aval <- to_abstract(x)
  if (is_rdata(aval)) aval$default_dtype else aval$dtype
}

# A traced box, with any R value in it committed to its default dtype. Anything
# that already has a dtype is returned unchanged.
commit_rdata_box <- function(x) {
  if (is_rdata_box(x)) {
    materialize_rdata(x, peek_dtype(x))
  } else {
    x
  }
}


# Whether a value has no dtype yet -- an RData box, or a bare R value that has
# not been boxed. Deliberately cheap: it runs once per argument of every traced
# primitive call.
has_no_dtype <- function(x) {
  is_rdata_box(x) || is_valid_r(x)
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
  sprintf("GraphLiteral(%s, %s, %s)", val, repr(x$aval$dtype), shape2string(x$aval$shape))
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
#' as part of a primitive call -- an input whose aval is an [`RData`].
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
#' @param rdata_types (`NULL | character()`)\cr
#'   One entry per input: the R storage type of an input the caller supplies as
#'   bare R data (`"double"`, `"integer"`, `"logical"`), and `NA` for one that
#'   arrives as an array and already has a data type. `NULL` when no input comes
#'   from R data, which is the common case. Together with the inputs\' own avals
#'   this says everything about how a call\'s arguments are uploaded: the aval
#'   gives the data type and shape, this gives the R type it is uploaded from.
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
  static_args_flat = NULL,
  rdata_types = NULL
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
  env$rdata_types <- rdata_types

  structure(env, class = "AnvlGraph")
}

graph_input_dtypes <- function(graph) {
  r_types <- graph$rdata_types
  if (is.null(r_types) || !any(!is.na(r_types))) {
    return(NULL)
  }
  ifelse(
    !is.na(r_types),
    vapply(graph$inputs, function(gval) as.character(gval$aval$dtype), character(1L)),
    NA_character_
  )
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
  # Bookkeeping for the R arguments, which are inputs whose data type is not
  # decided yet (an `RData` aval). Kept beside the descriptor rather than on a node
  # of it: none of it outlives the trace.
  #   rdata_mat:   input GraphValue -> list(dtype name -> GraphBox), the values
  #                the body built the argument at. One entry per dtype asked
  #                for, so asking twice reuses the value.
  #   rdata_outer: input GraphValue -> the enclosing trace's box for the same R
  #                argument, for an inline trace. Empty elsewhere.
  env$rdata_mat <- hashtab()
  env$rdata_outer <- hashtab()
  # One entry per input, set by finalize: the R storage type of an input the
  # caller supplies as bare R data, `NA` for one that arrives as an array.
  env$rdata_types <- NULL

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
    static_args_flat = descriptor$static_args_flat,
    rdata_types = descriptor$rdata_types
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

maybe_box_arrayish <- function(x, desc = .current_descriptor()) {
  if (is_graph_box(x)) {
    # An R value belongs to the graph it was written in, so one reaching
    # another graph has to commit before it can be captured there.
    if (is_rdata_box(x) && !identical(x$desc, desc)) {
      materialize_rdata(x, peek_dtype(x))
    }
    if (identical(x$desc, desc)) {
      return(x)
    }
    return(get_box_or_register_const(desc, x$gnode))
  }
  if (is_valid_r_lit(x) || is_valid_r_array(x)) {
    return(build_r_at(x, peek_dtype(x), desc))
  }
  if (is_anvl_array(x)) {
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
# - "inline": traced graph that will later be inlined into the parent
#   (gradient/value_and_gradient). Inputs that are not already boxed
#   are registered in the parent graph and then the inputs alias them,
#   which simplifies subsequent inlining. An R argument that is still open in
#   the parent trace stays open here too (see register_rdata_input()).
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
      x <- commit_rdata_box(x)
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
    if (is_rdata_box(x)) {
      # When we trace within a trace
      # Keep it open here too, so the traced body decides
      # which dtypes it is used at, exactly as it does under plain jit(). The
      # input slot is settled by finalize_inline_rdata_inputs().
      return(register_rdata_input(desc, x$gnode$aval, outer = x))
    }
    # \(x) gradient(f)(x)
    if (is_graph_box(x)) {
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
  if (is_rdata(x)) {
    # Bare R data passed to a jitted function. It takes an input slot like any
    # other argument -- the call has to supply the value -- but which dtype
    # that input has is only known once the body has used it, so the slot is
    # filled in by finalize_rdata_inputs().
    return(register_rdata_input(desc, x))
  }
  if (is_graph_box(x)) {
    x <- commit_rdata_box(x)
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

# Reserve `desc`'s next input slot for an R argument: an input like any other,
# except that its aval is an `RData` and so has no data type yet. The slot holds
# it until finalize_rdata_inputs() (or, for an inline trace,
# finalize_inline_rdata_inputs()) replaces it with the value the body
# materialized, which keeps the input order the same as the argument order (the
# caller supplies its inputs in that order).
#
# `outer` links an inline trace's input to the enclosing trace's box the
# argument came from. The value is fresh rather than shared so the inline body's
# materializations land in `desc` -- where transform_gradient() can
# differentiate the converts between them -- and cannot clobber the outer
# input's memo.
register_rdata_input <- function(desc, aval, outer = NULL) {
  gval <- GraphValue(aval)
  desc$inputs <- c(desc$inputs, list(gval))
  if (!is.null(outer)) {
    desc$rdata_outer[[gval]] <- outer
  }
  GraphBox(gval, desc)
}

# Which of a descriptor's inputs are R arguments whose data type is still open.
is_open_rdata_input <- function(gval) {
  is_rdata(gval$aval)
}

# modifies the descriptor in-place
# and finalizes the rdata inputs by:
# 1. Resolving the data type the input gets
# 2. Appending pre_calls with the conversions.
finalize_rdata_inputs <- function(desc) {
  inputs <- desc$inputs
  is_open <- vapply(inputs, is_open_rdata_input, logical(1L))
  if (!any(is_open)) {
    return(invisible(NULL))
  }
  r_types <- rep(NA_character_, length(inputs))
  pre_calls <- list()
  for (i in which(is_open)) {
    gval <- inputs[[i]]
    aval <- gval$aval
    mat <- rdata_mat(desc, gval)
    requested <- rdata_requested_dtypes(aval, mat)
    resolved <- resolve_upload_dtype(aval, requested)
    main <- mat[[resolved]] %||%
      GraphBox(GraphValue(AbstractArray(resolved, aval$shape)), desc)
    inputs[[i]] <- main$gnode
    r_types[[i]] <- aval$r_type
    for (other in setdiff(requested, resolved)) {
      # The invariance we need to uphold is we resolve the inputs in such a way, that this convert
      # always results in the same value
      pre_calls[[length(pre_calls) + 1L]] <- PrimitiveCall(
        primitive = prim_convert,
        inputs = list(main$gnode),
        params = list(dtype = as_dtype(other)),
        outputs = list(mat[[other]]$gnode)
      )
    }
  }
  desc$inputs <- inputs
  desc$rdata_types <- r_types
  desc$pre_calls <- c(desc$pre_calls, pre_calls)
  invisible(NULL)
}

# The dtypes a finished trace built `node`'s R value at directly. Only these
# can be uploaded (or serve as a graph input); the memo also holds the results
# of converting out of the value's category, which the program computes from
# one of these.
rdata_requested_dtypes <- function(aval, mat) {
  Filter(
    function(dt) rdata_builds_directly(aval$r_type, as_dtype(dt)),
    names(mat)
  )
}

# Finalize Rdata inputs of a sub-trace.
# E.g. used with gradent()
finalize_inline_rdata_inputs <- function(desc) {
  inputs <- desc$inputs
  is_open <- vapply(inputs, is_open_rdata_input, logical(1L))
  if (!any(is_open)) {
    return(invisible(NULL))
  }
  pre_calls <- list()
  add_convert <- function(input, dtype, output) {
    # The invariance we need to uphold is we resolve the inputs in such a way, that this convert
    # always results in the same value
    pre_calls[[length(pre_calls) + 1L]] <<- PrimitiveCall(
      primitive = prim_convert,
      inputs = list(input),
      params = list(dtype = as_dtype(dtype)),
      outputs = list(output)
    )
  }
  for (i in which(is_open)) {
    gval <- inputs[[i]]
    aval <- gval$aval
    mat <- rdata_mat(desc, gval)
    requested <- rdata_requested_dtypes(aval, mat)
    resolved <- resolve_upload_dtype(aval, requested)
    outer <- desc$rdata_outer[[gval]]
    local_box <- mat[[resolved]]
    # In the examples below, `x` is `nv_scalar(1, "f64")` and `b` is the R
    # value this iteration settles.
    if (is.null(local_box)) {
      # The sub-trace never built at the resolved data type
      # (e.g., if sub-trace uses input 1 at f16 and bf16, resolved would be f32, i.e. the one that
      # can hold both, even though nobody requested it)
      main <- materialize_rdata(outer, as_dtype(resolved))$gnode
    } else if (is.null(rdata_mat(outer$desc, outer$gnode)[[resolved]])) {
      # The value at `resolved` exists in this graph and nowhere else: hand
      # the gval itself up, so the enclosing trace's own finalize defines it
      # (as the upload input, or a convert from it) ahead of this graph's
      # calls, and later uses there reuse it.
      # e.g. the body builds `q` at `f64`, the enclosing trace never uses `b`:
      #   jit(\(a, b) gradient(\(p, q) p * q, wrt = "p")(a, b))(x, 2)
      main <- local_box$gnode
      rdata_mat_set(outer$desc, outer$gnode, resolved, register_gval(outer$desc, main))
    } else {
      # The enclosing trace built `resolved` too, so the body's gval is a
      # second value for it: take the outer one as the input and define the
      # body's from it.
      # e.g. `a + b` builds `b` at `f64` before the body builds its own:
      #   jit(\(a, b) { w <- a + b; gradient(\(p, q) p * q, wrt = "p")(a, b) })(x, 2)
      main <- materialize_rdata(outer, as_dtype(resolved))$gnode
      add_convert(main, resolved, local_box$gnode)
    }
    inputs[[i]] <- main
    for (other in setdiff(requested, resolved)) {
      add_convert(main, other, mat[[other]]$gnode)
    }
  }
  desc$inputs <- inputs
  desc$pre_calls <- c(desc$pre_calls, pre_calls)
  invisible(NULL)
}

# What a dtype can hold, as (precision, range): a float's mantissa and exponent
# width, an integer's width and whether it is signed -- an R integer is signed,
# so an unsigned dtype of the same width holds less of it. Used only to compare
# dtypes of one category, which is the only comparison that means anything.
dtype_capacity <- function(dtype) {
  switch(
    as.character(dtype),
    f64 = c(52L, 11L),
    f32 = c(23L, 8L),
    f16 = c(10L, 5L),
    bf16 = c(7L, 8L),
    c(dtype_width(dtype), if (is_dtype_int(dtype)) 1L else 0L)
  )
}

# Whether building an R value at `dtype` and converting to `other` gives what
# building it at `other` directly would -- i.e. whether `dtype` holds every
# value `other` can express.
dtype_holds <- function(dtype, other) {
  all(dtype_capacity(dtype) >= dtype_capacity(other))
}

# The dtypes an R value of this storage type can be built at, narrowest first.
# `resolve_upload_dtype()` searches these when no dtype the trace asked for holds
# the others, so the upload widens only as far as it must.
rdata_build_candidates <- function(r_type) {
  switch(
    r_type,
    double = c("f16", "bf16", "f32", "f64"),
    # An R integer is signed and is not built below 32 bits
    # (`rdata_builds_directly()`), so these are all of them.
    integer = c("i32", "i64"),
    logical = "bool",
    cli_abort("No build candidates for R type {.val {r_type}}")
  )
}

# The single dtype an R argument is uploaded at, given every dtype the trace
# built it at. Those are all in the value's own category (materialize_rdata()
# converts out of it inside the program instead), so one of them can serve every
# use site: the upload has to *hold* them all, and each site then converts down
# from it, rounding exactly once.
#
# Not simply the widest. `f16` and `bf16` are both 16 bits and neither holds the
# other -- `f16` has three more mantissa bits, `bf16` a far wider exponent. When
# no dtype the trace asked for holds them all, the narrowest one of the value's
# category that does is used (`f32` for those two), so a program with no `f64`
# in it does not acquire one here.
resolve_upload_dtype <- function(aval, requested) {
  if (!length(requested)) {
    return(as.character(aval$default_dtype))
  }
  dtypes <- lapply(requested, as_dtype)
  holds_all <- vapply(dtypes, function(d) all(vapply(dtypes, dtype_holds, logical(1L), dtype = d)), logical(1L))
  if (!any(holds_all)) {
    # None of them holds the others, so the upload has to widen -- but only as
    # far as it takes. The natural dtype would always do (`f64` holds every
    # float), and it is what the value would be built at on its own; taking it
    # here would put an `f64` input into a program that has no other `f64` in
    # it, purely because one argument was asked for at two narrow floats.
    holds <- function(cand) all(vapply(dtypes, dtype_holds, logical(1L), dtype = as_dtype(cand)))
    narrowest <- Find(holds, rdata_build_candidates(aval$r_type))
    return(narrowest %||% as.character(rdata_natural_dtype(aval$r_type)))
  }
  # Several candidates hold them all only when they are equivalent; ordering by
  # name keeps the choice independent of the order the trace happened to ask in.
  names <- vapply(dtypes[holds_all], as.character, character(1L))
  sort(names)[[1L]]
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
  outputs_flat <- lapply(output[[2L]], function(x) commit_rdata_box(maybe_box_arrayish(x)))

  desc$out_tree <- out_tree
  desc$outputs <- lapply(outputs_flat, \(x) x$gnode)
  # Where this trace's still-open R arguments end up, which is what the two
  # differ in. A toplevel trace owns them: they become its own inputs, at the
  # dtype the caller uploads them at. An inline trace does not -- they belong to
  # the enclosing trace and stay open there, so the input is handed back up to
  # it and only the converts between dtypes stay here, where
  # transform_gradient() differentiates them. A sub-graph has none to settle:
  # its R values commit when `maybe_box_input()` builds the parameter slots.
  # We might
  if (mode == "toplevel") {
    # Standard case:
    finalize_rdata_inputs(desc)
  } else if (mode == "inline") {
    # This e.g. happens in:  jit(\(x, y) gradient(\(x, y) x * y, wrt = "x")(x, y))(nv_scalar(1, "f64"), 2)
    # There, the data type at which pi is materialized depends on the result of the sub-trace
    # but it needs to be regsitered for the toplevel trace
    finalize_inline_rdata_inputs(desc)
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
    # Commit R values to their default dtype, which happens when no promotion rule
    # materialized them (default behavior)
    gnode <- commit_rdata_box(maybe_box_arrayish(args[[i]], desc))$gnode
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
