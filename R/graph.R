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
  sprintf("GraphLiteral(%s, %s, %s)", val, as.character(x$aval$dtype), shape2string(x$aval$shape))
}

#' @export
print.GraphLiteral <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

#' @title Graph Node
#' @description
#' Virtual base class for nodes in an [`AnvlGraph`].
#' Is either a [`GraphValue`] or a [`GraphLiteral`].
#' Cannot be instantiated directly - use [`GraphValue()`] or [`GraphLiteral()`] instead.
#' @name GraphNode
NULL

#' @title Graph Statement
#' @description
#' One statement of an [`AnvlGraph`]: a primitive applied to inputs, with its
#' results assigned to outputs.
#' @param primitive (`AnvlPrimitiveDef`)\cr
#'   The function.
#' @param inputs (`list(GraphValue)`)\cr
#'   The (array) inputs to the primitive.
#' @param params (`list(<any>)`)\cr
#'   The (static) parameters of the function call.
#' @param outputs (`list(GraphValue)`)\cr
#'   The (array) outputs of the primitive.
#' @return (`GraphStatement`)
#' @export
GraphStatement <- function(primitive, inputs, params, outputs) {
  if (inherits(primitive, "AnvlPrimitive")) {
    primitive <- attr(primitive, "definition")
  }
  # hot-path constructor: no input validation
  structure(
    list(
      primitive = primitive,
      inputs = inputs,
      params = params,
      outputs = outputs
    ),
    class = "GraphStatement"
  )
}

#' @title Graph of Statements
#'
#' @description
#' Computational graph consisting exclusively of statements that apply primitives.
#' This is a mutable class.
#'
#' An `AnvlGraph` is usually created by tracing a function with
#' [`trace_fn()`], which records each primitive call as a [`GraphStatement`]
#' into a [`GraphDescriptor`] and converts it into an `AnvlGraph` at the end. The
#' graph is then lowered, e.g. with [`stablehlo()`], and compiled.
#'
#' @param statements (`list(GraphStatement)`)\cr
#'   The statements that make up the graph.
#' @param in_tree (`NULL` | [`RTree`][pjrt::build_tree])\cr
#'   The tree of inputs. May contain leaves for both array inputs and static
#'   (non-array) arguments. Only the array leaves correspond to entries in
#'   `inputs`; use `is_static_flat` to distinguish them.
#' @param out_tree (`NULL` | [`RTree`][pjrt::build_tree])\cr
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
#'   from R data, which is the common case. Together with the inputs' own avals
#'   this says everything about how a call's arguments are uploaded: the aval
#'   gives the data type and shape, this gives the R type it is uploaded from.
#' @return (`AnvlGraph`)
#' @examplesIf pjrt::plugins_downloaded()
#' # the inputs are %x1 and %x2; each line is one statement, and `sum`
#' # shows its parameters in brackets
#' graph <- trace_fn(function(x, y) {
#'   nv_sum(x * y)
#' }, list(x = nv_aval("f32", c(2, 3)), y = nv_aval("f32", c(2, 3))))
#' graph
#' graph$inputs
#' graph$outputs
#'
#' # an array the function closes over becomes the constant %c1, and the R
#' # value `2` the literal `2:f32`
#' w <- nv_array(c(1, 2), dtype = "f32")
#' graph <- trace_fn(function(x) x + w * 2, list(x = nv_aval("f32", 2)))
#' graph
#' graph$constants
#'
#' # several outputs are returned together; `out_tree` records their structure
#' graph <- trace_fn(function(x) list(a = x, b = nv_exp(x)), list(x = nv_aval("f32", c())))
#' graph
#' graph$out_tree
#' @keywords internal
# @export
AnvlGraph <- function(
  statements = list(),
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
  env$statements <- statements
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

#' @title Graph Descriptor
#' @description
#' The in-progress representation of an [`AnvlGraph`] during tracing. This is
#' a mutable class.
#'
#' While [`trace_fn()`] runs a function, every primitive call is recorded as a
#' [`GraphStatement`] into the current descriptor (see [`graph_desc_add()`] and
#' [`current_descriptor()`]). The descriptor also does the book-keeping the
#' trace needs: which [`GraphValue`] an `AnvlArray` or [`GraphBox`] stands
#' for, the constants and devices encountered, and the default data types the
#' trace is pinned to. Once tracing finishes, it is converted to an
#' [`AnvlGraph`], which only keeps what is needed to lower and run the
#' program.
#' @param statements (`list(GraphStatement)`)\cr
#'   The statements that make up the graph.
#' @param array_to_gval (`hashtab`)\cr
#'   Mapping: `AnvlArray` -> `GraphValue`
#' @param gval_to_box (`hashtab`)\cr
#'   Mapping: `GraphValue` -> `GraphBox`
#' @param constants (`list(GraphValue)`)\cr
#'   The constants of the graph.
#' @param in_tree (`NULL` | [`RTree`][pjrt::build_tree])\cr
#'   The tree of inputs. May contain leaves for both array inputs and static
#'   (non-array) arguments. Only the array leaves correspond to entries in
#'   `inputs`; use `is_static_flat` to distinguish them.
#' @param out_tree (`NULL` | [`RTree`][pjrt::build_tree])\cr
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
#' @param default_dtypes (`NULL` | `list(float, int)`)\cr
#'   The data types every R value in this trace materializes at when nothing
#'   else decides one (see [`default_dtypes()`]).
#' @param backend (`character(1)`)\cr
#'   The backend this trace is compiled for. Required: it decides which entry
#'   of the `anvl.default_dtypes` option applies to the trace, so switching the
#'   active backend inside a traced body changes nothing.
#'   [`local_descriptor()`] fills it in from [`active_backend()`], so only a
#'   direct call has to name it.
#' @param devices (`list()`)\cr
#'   Devices encountered during tracing: the device of every concrete array
#'   registered in the graph, plus the ones declared by [`graph_desc_add()`].
#' @return (`GraphDescriptor`)
#' @export
GraphDescriptor <- function(
  statements = list(),
  array_to_gval = NULL,
  gval_to_box = NULL,
  constants = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  is_static_flat = NULL,
  static_args_flat = NULL,
  devices = character(),
  default_dtypes = NULL,
  backend
) {
  # Use an environment for reference semantics (mutable)
  env <- new.env(parent = emptyenv())
  # `statements` accumulates one entry per traced primitive. A fastqueue gives
  # amortised-O(1) append; growing an R list here (`env$statements[[n]] <- x` or
  # `c(env$statements, x)`) is copy-on-modify and would make tracing O(n^2).
  env$statements <- fastmap::fastqueue()
  if (length(statements)) {
    env$statements$madd(.list = statements)
  }
  env$array_to_gval <- array_to_gval %||% hashtab()
  env$gval_to_box <- gval_to_box %||% hashtab()
  env$constants <- constants
  env$in_tree <- in_tree
  env$out_tree <- out_tree
  env$inputs <- inputs
  env$outputs <- outputs
  env$is_static_flat <- is_static_flat
  env$static_args_flat <- static_args_flat
  env$devices <- devices
  env$default_dtypes <- default_dtypes
  env$backend <- backend
  # Calls that have to run before everything else, because they only depend on
  # the graph's inputs: the converts finalize_rdata_inputs() adds for an R
  # argument that one program used at more than one dtype.
  env$pre_statements <- list()
  # Bookkeeping for the R arguments, which are inputs whose data type is not
  # decided yet (an `RData` aval). Kept beside the descriptor rather than on a node
  # of it: none of it outlives the trace.
  #   rdata_mat:   input GraphValue -> list(dtype name -> GraphBox), the values
  #                the body built the argument at. One entry per dtype asked
  #                for, so asking twice reuses the value.
  env$rdata_mat <- hashtab()
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
    statements = c(descriptor$pre_statements, descriptor$statements$as_list()),
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
#' Wraps a [`GraphNode`] during graph construction (tracing).
#' When a function is traced via [`trace_fn()`], each intermediate array
#' value is represented as a `GraphBox`.
#' It also contains an associated [`GraphDescriptor`] in which the node "lives".
#'
#' @param gnode ([`GraphNode`])\cr
#'   The graph node -- either a [`GraphValue`] or a [`GraphLiteral`].
#' @param desc ([`GraphDescriptor`])\cr
#'   The descriptor of the graph being built.
#' @return (`GraphBox`)
#'
#' @seealso [trace_fn()], [jit()]
#' @export
GraphBox <- function(gnode, desc) {
  # hot-path constructor: no input validation
  structure(
    list(gnode = gnode, desc = desc),
    class = "GraphBox"
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

maybe_box_arrayish <- function(x, desc = current_descriptor()) {
  if (is_graph_box(x)) {
    # An R value belongs to the graph it was written in, so one reaching
    # another graph has to materialize before it can be captured there.
    if (is_rdata_box(x) && !identical(x$desc, desc)) {
      x <- materialize_rdata(x, peek_dtype(x))
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

# Called only by trace_fn() to wire up each flat arg as an input of `desc`,
# the same way for jit's outermost trace and for a trace inside another one:
# the sub-graphs of a higher-order primitive (prim_if/prim_while/...), and the
# function gradient() differentiates.
#
# Each arrayish arg becomes a fresh input gval. In a trace inside another one it
# is bound only later to a box of the parent: by the call's operands, or by
# gradient() replaying the graph into the parent. An R value with no data type
# yet -- bare R data jit was called with, which the dispatcher describes as an
# `RData`, or an argument of a trace further out that has not been given a data
# type -- becomes an input whose data type the body decides, as
# finalize_rdata_inputs() settles. Whoever binds the input then materializes the
# outer value at that data type, which records the use in the descriptor the
# value belongs to (see `materialize_rdata()`); that descriptor is finalized
# last, so it sees every use, however deeply nested.
#
# Anything else, a bare R value included, passes through as a static arg: for
# jit it is one of the args declared static (R data it was called with arrives
# as an `RData`), a primitive whose sub-graph takes its operands (prim_while's
# state, prim_scan's carry) materializes them itself, and gradient() does not
# know which of its args are static.
maybe_box_input <- function(x, desc) {
  if (is_anvl_array(x)) {
    desc$devices <- c(desc$devices, placement_device(x))
    return(register_input(desc, GraphValue(aval = to_abstract(x, pure = TRUE))))
  }
  # An open R value keeps its `RData` aval, so it stays open here too.
  if (is_graph_box(x)) {
    return(register_input(desc, GraphValue(aval = abstract_aval(x$gnode$aval))))
  }
  # An `RData` from the dispatcher is an abstract array too; prim_reduce() and
  # prim_scatter() trace their scalar functions with avals.
  if (is_abstract_array(x)) {
    return(register_input(desc, GraphValue(aval = x)))
  }
  x
}

# Strip data from a (possibly concrete) array aval, returning a pure
# AbstractArray with the same dtype and shape.
abstract_aval <- function(aval) {
  if (is_concrete_array(aval)) {
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
    desc$devices <- c(desc$devices, placement_device(x))
    gval <- desc$array_to_gval[[x]]
    if (!is.null(gval)) {
      return(desc$gval_to_box[[gval]])
    }
    gval <- GraphValue(aval = ConcreteArray(x))
    desc$array_to_gval[[x]] <- gval
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
  # Every graph that closes over an array mints a GraphValue of its own for
  # it, so an array `desc` already holds is matched by itself: the box returned
  # may then be of another GraphValue than `x`.
  if (is_concrete_array(x$aval)) {
    known <- desc$array_to_gval[[x$aval$data]]
    if (!is.null(known)) {
      return(desc$gval_to_box[[known]])
    }
  }

  # Now, we create the new box and register it, so if we see it again, we can return it immediately.
  new_box <- GraphBox(x, desc)

  if (is_concrete_array(x$aval)) {
    desc$array_to_gval[[x$aval$data]] <- x
  }
  desc$gval_to_box[[x]] <- new_box
  desc$constants <- c(desc$constants, list(x))
  return(new_box)
}

# A value a higher-order primitive passes both to its sub-graphs, as an input,
# and to its call, as an operand -- `prim_while()`'s state, `prim_scan()`'s
# carry -- boxed in `desc` with a data type. Tracing the sub-graphs would leave
# a bare R value static, and one with no data type yet open in each of them
# separately, so it materializes at its default here, once for all of them.
materialize_operand <- function(x, desc) {
  materialize_rdata_box(maybe_box_arrayish(x, desc))
}

# Makes the sub-graphs `graphs` of one higher-order call pure functions of
# their inputs: what they captured from outside (recorded as their `constants`
# while traced) becomes trailing inputs, the same ones in the same order for
# every graph. Modifies the graphs
# in place and returns the boxes in `desc` the call passes for the captures,
# after its own operands; its `n_captures` param is their number.
purify_subgraphs <- function(desc, graphs) {
  # Collect the captures of all graphs, each once. `capture_index_of_gval`
  # maps a captured GraphValue to its position among them. A value computed by
  # an enclosing graph is captured as that graph's own GraphValue, so sibling
  # sub-graphs share it and identity suffices. A closed-over array is not:
  # every sub-graph mints a GraphValue of its own for it, so an array is
  # matched by itself, via `capture_index_of_array`.
  capture_index_of_gval <- hashtab()
  capture_index_of_array <- hashtab()
  captures <- list()
  for (graph in graphs) {
    for (gval in graph$constants) {
      if (!is.null(capture_index_of_gval[[gval]])) {
        next
      }
      array <- if (is_concrete_array(gval$aval)) gval$aval$data
      index <- if (!is.null(array)) capture_index_of_array[[array]]
      if (is.null(index)) {
        captures[[length(captures) + 1L]] <- gval
        index <- length(captures)
        if (!is.null(array)) {
          capture_index_of_array[[array]] <- index
        }
      }
      capture_index_of_gval[[gval]] <- index
    }
  }
  for (graph in graphs) {
    # Every graph gets an input per capture, also for those only another graph
    # uses, so that all of them take the call's operands the same way.
    fresh <- lapply(captures, function(gval) GraphValue(aval = abstract_aval(gval$aval)))
    # Its calls read that input in place of the outer value, which makes the
    # graph a pure function of its inputs.
    map <- hashtab()
    for (gval in graph$constants) {
      map[[gval]] <- fresh[[capture_index_of_gval[[gval]]]]
    }
    substitute_gnodes(graph, map)
    graph$inputs <- c(graph$inputs, fresh)
    graph$constants <- list()
  }
  # The outer values as boxes of `desc`. If `desc` is itself a sub-graph being
  # traced, this is where it captures them in turn.
  lapply(captures, function(gval) get_box_or_register_const(desc, gval))
}

# Readies `graph`, the sub-graph of `prim_reduce()` or `prim_scatter()`, for
# `purify_subgraphs()`. It lowers to a StableHLO region that, unlike an `if`
# or `while` region, cannot read a value of the function around it, so it must
# not capture anything. A scalar array it closes over is inlined as a literal
# of its own; any other value it closes over is refused. `fn` names the
# argument `graph` was traced from, for the error.
inline_region_captures <- function(graph, fn) {
  map <- hashtab()
  for (gval in graph$constants) {
    if (is_concrete_array(gval$aval) && nelts(gval$aval) == 1L) {
      map[[gval]] <- GraphLiteral(LiteralArray(
        gval$aval$data,
        shape = shape(gval$aval),
        dtype = dtype(gval$aval)
      ))
    }
  }
  substitute_gnodes(graph, map)
  graph$constants <- Filter(\(gval) is.null(map[[gval]]), graph$constants)
  if (length(graph$constants)) {
    cli_abort(c(
      "{.arg {fn}} must not close over a traced value or a non-scalar array.",
      i = "Only scalar arrays and R literals can be used inside {.arg {fn}}."
    ))
  }
  invisible(graph)
}

# Replaces, in place, every node of `graph`'s calls and outputs that `map` has
# an entry for. Nested sub-graphs are closed, so they reach an outer value only
# through an operand of their call, and this graph's calls are all there is to
# rewrite.
substitute_gnodes <- function(graph, map) {
  sub <- function(g) if (is_graph_literal(g)) g else map[[g]] %||% g
  graph$statements <- lapply(graph$statements, function(call) {
    call$inputs <- lapply(call$inputs, sub)
    call
  })
  graph$outputs <- lapply(graph$outputs, sub)
  invisible(graph)
}

register_inputs <- function(desc, inputs) {
  for (input in inputs) {
    register_input(desc, input)
  }
}

match_args_to_formals <- function(f, args) {
  g <- function() {
    as.list(match.call()[-1L])
  }
  formals(g) <- formals(f)
  do.call(g, args)
}

# Points `e`'s call at the primitive the marker names while the error is on its
# way out, once: a sub-graph trace names it before restoring the marker to the
# higher-order primitive that traced it, which would otherwise take the blame
# at the top level. With no primitive marked, the call is left as raised.
name_failing_primitive <- function(e) {
  if (isTRUE(e$anvl_primitive_named)) {
    return(e)
  }
  prim <- globals[["INFER_PRIMITIVE"]]
  if (!is.null(prim)) {
    e$call <- print_call_repr(prim)
  }
  e$anvl_primitive_named <- TRUE
  e
}

#' @title Trace an R Function into a Graph
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
#' @param args_flat (`list`)\cr
#'   Flattened arguments. Must be accompanied by `in_tree`.
#' @param in_tree ([`RTree`][pjrt::build_tree])\cr
#'   Tree structure describing how `args_flat` maps back to `f`'s arguments.
#' @template param_optimize
#' @return ([`AnvlGraph`])
#'   Contains the traced operations.
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
  args_flat = NULL,
  in_tree = NULL,
  optimize = FALSE
) {
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

  # box arrays and add them as inputs to the current graph
  inputs_flat <- lapply(args_flat, maybe_box_input, desc = desc)
  # Track which flat args are static (non-array) values vs. graph inputs
  desc$is_static_flat <- vapply(inputs_flat, Negate(is_graph_box), logical(1L))
  # A higher-order primitive traces its sub-graphs here and then goes on to
  # check them, so the primitive it named on the way in has to survive the
  # sub-trace: every primitive *inside* the sub-graph names itself and clears
  # the marker again on its way out. An error out of the sub-graph restores it
  # too, but is named first, so it keeps the primitive that raised it. The
  # outermost trace starts from no marker, whatever an earlier error left.
  prim <- if (!is.null(parent_desc)) globals[["INFER_PRIMITIVE"]]
  globals[["INFER_PRIMITIVE"]] <- prim
  output <- tryCatch(
    do.call(f_flat, inputs_flat),
    error = function(e) {
      e <- name_failing_primitive(e)
      globals[["INFER_PRIMITIVE"]] <- prim
      rlang::cnd_signal(e)
    }
  )
  globals[["INFER_PRIMITIVE"]] <- prim

  out_tree <- output[[1L]]
  # function() x; -> output can be an closed-over constant
  outputs_flat <- lapply(output[[2L]], function(x) materialize_rdata_box(maybe_box_arrayish(x)))

  desc$out_tree <- out_tree
  desc$outputs <- lapply(outputs_flat, \(x) x$gnode)
  # A still-open R argument becomes an input at the data type the body settled
  # it at. For a toplevel trace that is the dtype the caller uploads it at; for
  # a sub-graph, the one whoever binds the input materializes the outer value
  # at -- in `gradient()`, the finalize of the trace the value belongs to then
  # takes it into account like any other use there.
  finalize_rdata_inputs(desc)
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

#' @title Get the Current Graph
#' @description
#' Get the current graph being built.
#' @param silent (`logical(1)`)\cr
#'   Whether to return `NULL` if no graph is currently being built (as opposed to aborting).
#' @return ([`GraphDescriptor`] | `NULL`)\cr
#'   `NULL` only when `silent = TRUE` and no graph is being built.
#' @export
current_descriptor <- function(silent = FALSE) {
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

#' @title Create a Graph
#' @description
#' Creates a new [`GraphDescriptor`] which is afterwards accessible via [`current_descriptor()`].
#' The graph is automatically removed when exiting the current scope.
#' After the graph is either cleaned up automatically (by exiting the scope)
#' or finalized, the previously built graph is restored,
#' i.e., accessible via [`current_descriptor()`].
#'
#' @param envir (`environment`)\cr
#'   Environment where exit handler will be registered for cleaning up the
#'   [`GraphDescriptor`] if it was not returned yet.
#' @param ... (`any`)\cr
#'   Additional arguments to pass to the [`GraphDescriptor`] constructor.
#' @return ([`GraphDescriptor`])
#' @export
local_descriptor <- function(..., envir = parent.frame()) {
  if (identical(envir, globalenv())) {
    # lingering global descriptors interfere with graph tracing
    cli_abort("Don't run local_descriptor in the global environment")
  }

  args <- list(...)
  # assumes that backend does not change during a trace.
  # If this happens, we get undefined behavior.
  args$backend <- args$backend %||% active_backend()
  args$default_dtypes <- args$default_dtypes %||% current_default_dtypes()
  desc <- do.call(GraphDescriptor, args)
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

#' @title Add a Statement to a Graph Descriptor
#' @description
#' Record a call of a primitive as a [`GraphStatement`] in a graph descriptor. Inside a primitive body created
#' with [`new_primitive()`], pass the lexically-bound `self` as the primitive
#' argument.
#' @param primitive ([`AnvlPrimitiveDef`] | [`AnvlPrimitive`])\cr
#'   The primitive the statement applies. An `AnvlPrimitive` is accepted and
#'   unwrapped to its underlying `AnvlPrimitiveDef`.
#' @param args (`list` of [`arrayish`])\cr
#'   The arguments to the primitive: [`GraphBox`]es, [`AnvlArray`]s (registered
#'   as constants of the graph) or R values (materialized at their default data
#'   type).
#' @param params (`list`)\cr
#'   The parameters to the primitive.
#' @param infer_fn (`function`)\cr
#'   The inference function to use.
#'   Must output a list of [`AbstractArray`]s.
#' @param desc ([`GraphDescriptor`] | `NULL`)\cr
#'   The graph descriptor to add the statement to.
#'   Uses the [current descriptor][current_descriptor] if `NULL`.
#' @param device (`NULL` | `character(1)` | device object)\cr
#'   The device the call places its result on, for a primitive that constructs
#'   an array out of nothing (e.g. [`prim_fill()`], [`prim_iota()`]) and so has
#'   no operand to carry one. It is declared to `desc`, where it counts like
#'   the device of an array input to the same trace: it decides what that
#'   program is compiled for, and disagreeing with another device in it is an
#'   error. Every other primitive takes its device from its operands and leaves
#'   this `NULL`.
#' @return (`list` of [`GraphBox`])
#' @export
graph_desc_add <- function(primitive, args, params = list(), infer_fn, desc = NULL, device = NULL) {
  desc <- desc %||% current_descriptor(silent = TRUE)
  if (!is.null(device)) {
    desc$devices <- c(desc$devices, nv_device(device))
  }
  if (inherits(primitive, "AnvlPrimitive")) {
    primitive <- attr(primitive, "definition")
  }

  # Box each input and pull out its gnode + aval in one pass (`gnodes_in`
  # unnamed for the GraphStatement; `avals_in` keeps arg names for infer_fn).
  n_in <- length(args)
  gnodes_in <- vector("list", n_in)
  avals_in <- vector("list", n_in)
  for (i in seq_len(n_in)) {
    # Materialize R values at their default dtype, which happens when no
    # promotion rule materialized them (default behavior)
    gnode <- materialize_rdata_box(maybe_box_arrayish(args[[i]], desc))$gnode
    gnodes_in[[i]] <- gnode
    avals_in[[i]] <- gnode$aval
  }
  names(avals_in) <- names(args)
  # The primitive under way is named by its wrapper, on the way in; this clears
  # it again once inference has passed, so that a later error somewhere else in
  # the traced function is not attributed to the last primitive that ran.
  ats_out <- do.call(infer_fn, c(avals_in, params))
  globals[["INFER_PRIMITIVE"]] <- NULL
  # An output is computed, whatever its inputs were: an inference rule that
  # hands an input's aval back (`infer_generic_biv()` returns `lhs`) must not
  # make `sin(y)` of a closed-over `y` look like `y` itself. Only a constant's
  # node carries a `ConcreteArray`.
  gvals_out <- lapply(ats_out, \(aval) GraphValue(abstract_aval(aval)))
  call <- GraphStatement(primitive, gnodes_in, params, gvals_out)
  desc$statements$add(call)
  lapply(gvals_out, register_gval, desc = desc)
}

# A primitive is named for the `prim_*()` that exports it, so the call an error
# reports is that name with the prefix put back on.
print_call_repr <- function(prim) {
  rlang::exec(call, paste0("prim_", prim$name))
}
