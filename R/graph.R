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
  checkmate::assert_class(aval, "AbstractArray")

  # Use an environment for reference semantics (mutable)
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
  checkmate::assert_class(aval, "LiteralArray")

  # Use an environment for reference semantics (mutable)
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
  sprintf("GraphLiteral(%s, %s, %s)", val, dtype2string(x$aval$dtype, x$aval$ambiguous), shape2string(x$aval$shape))
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
  checkmate::assert_class(primitive, "AnvlPrimitive")
  checkmate::assert_list(inputs, types = c("GraphValue", "GraphLiteral"))
  checkmate::assert_list(params)
  checkmate::assert_list(outputs, c("GraphValue", "GraphLiteral"))

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
  env$calls <- calls
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
ambiguous.GraphValue <- function(x, ...) {
  x$aval$ambiguous
}

#' @export
shape.GraphLiteral <- function(x, ...) {
  shape(x$aval)
}

#' @export
dtype.GraphLiteral <- function(x, ...) {
  x$aval$dtype
}

#' @export
ambiguous.GraphLiteral <- function(x, ...) {
  x$aval$ambiguous
}


is_graph_descriptor <- function(x) {
  inherits(x, "GraphDescriptor")
}

descriptor_to_graph <- function(descriptor) {
  graph <- AnvlGraph(
    calls = descriptor$calls,
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
#' - [`ndims()`][tengen::ndims]
#' - [`ambiguous()`]
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
  if (!is_graph_node(gnode)) {
    cli_abort("gnode must be a GraphValue or GraphLiteral")
  }
  checkmate::assert_class(desc, "GraphDescriptor")

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
ambiguous.GraphBox <- function(x, ...) {
  ambiguous(x$gnode)
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

maybe_box_arrayish <- function(x) {
  current_desc <- .current_descriptor()
  if (is_graph_box(x)) {
    if (identical(x$desc, current_desc)) {
      return(x)
    }
    gval <- x$gnode
    return(get_box_or_register_const(current_desc, gval))
  }
  if (is_valid_r_array(x)) {
    # Materialize R arrays as plain-backend AnvlArrays so they can be
    # registered as named constants in the current graph.
    x <- nv_array(x, ambiguous = !is.logical(x))
  }
  if (is_anvl_array(x) || is_valid_r_lit(x)) {
    return(get_box_or_register_const(current_desc, x))
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
      x <- nv_scalar(x, ambiguous = !is.logical(x))
    } else if (is_valid_r_array(x)) {
      x <- nv_array(x, ambiguous = !is.logical(x))
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
  if (is_graph_box(x)) {
    return(register_input(desc, x$gnode))
  }
  if (is_abstract_array(x)) {
    gval <- GraphValue(aval = x)
    return(register_input(desc, gval))
  }
  x
}

# Strip data from a (possibly concrete) array aval, returning a pure
# AbstractArray with the same dtype/shape/ambiguity.
abstract_aval <- function(aval) {
  if (is_concrete_tensor(aval)) {
    AbstractArray(dtype = aval$dtype, shape = aval$shape, ambiguous = aval$ambiguous)
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
  if (!is_graph_descriptor(desc)) {
    cli_abort("Internal error: trying to register a gval in a non-graph descriptor")
  }
  if (!is_graph_value(x)) {
    cli_abort("Internal error: trying to register an invalid gval")
  }
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
  if (!is_graph_descriptor(desc)) {
    cli_abort("Internal error: trying to register a constant in a non-graph descriptor")
  }
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
    ambiguous <- !is.logical(x)
    gval <- GraphLiteral(LiteralArray(x, shape = integer(), ambiguous = ambiguous))
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
#' @return An [`AnvlGraph`] containing the traced operations.
#' @seealso [`stablehlo()`] to lower the graph, [`jit()`] / [`xla()`] for end-to-end
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
  in_tree = NULL
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
  f_flat <- pjrt::flatten_fun(f, in_node = in_tree)
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
  output <- do.call(f_flat, inputs_flat)

  out_tree <- output[[1L]]
  # function() x; -> output can be an closed-over constant
  outputs_flat <- lapply(output[[2L]], maybe_box_arrayish)

  desc$out_tree <- out_tree
  desc$outputs <- lapply(outputs_flat, \(x) x$gnode)
  if (!is.null(desc$is_static_flat) && isTRUE(any(desc$is_static_flat))) {
    desc$static_args_flat <- args_flat[desc$is_static_flat]
  } else {
    desc$static_args_flat <- NULL
  }

  graph <- descriptor_to_graph(desc)
  return(graph)
}

is_graph_node <- function(x) {
  is_graph_value(x) || is_graph_literal(x)
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
  !is.null(.current_descriptor(silent = TRUE))
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
  checkmate::assert_class(primitive, "AnvlPrimitive")

  boxes_in <- lapply(args, maybe_box_arrayish)
  gnodes_in <- unname(lapply(boxes_in, \(box) box$gnode))
  avals_in <- lapply(boxes_in, \(box) box$gnode$aval)
  ats_out <- tryCatch(
    {
      rlang::exec(infer_fn, !!!c(avals_in, params))
    },
    error = function(e) {
      e$call <- print_call_repr(primitive)
      e <- stablehlo::to_one_based(e)
      rlang::cnd_signal(e)
    }
  )
  gvals_out <- lapply(ats_out, GraphValue)
  call <- PrimitiveCall(primitive, gnodes_in, params, gvals_out)
  desc$calls <- c(desc$calls, list(call))
  lapply(gvals_out, register_gval, desc = desc)
}

print_call_repr <- function(prim) {
  rlang::exec(call, paste0("prim_", prim$name))
}


inline_graph_into_desc <- function(desc, graph) {
  # By contract, `graph` was produced by `trace_fn(..., mode = "inline")` so
  # every input gval was registered in `desc` at trace time and is already
  # known to the parent. Only sub-graph constants (closed-over values
  # registered via `maybe_box_arrayish`) still need to be propagated up.
  register_consts(desc, graph$constants)

  desc$calls <- c(desc$calls, graph$calls)

  gvals_out_flat <- graph$outputs
  boxes_out_flat <- lapply(gvals_out_flat, GraphBox, desc)
  unflatten(graph$out_tree, boxes_out_flat)
}
