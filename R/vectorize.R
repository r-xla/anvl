#' @title Vectorize a Function
#' @description
#' Return a new function that maps `f` over an axis of its arguments, like
#' [base::Vectorize()]. Unlike `Vectorize()`, it does not loop: the returned
#' function traces `f` once and turns every operation of it into one that works
#' on the whole batch, so `f` can be written for a single instance and still
#' run as one program over all of them.
#'
#' The arguments in `args` are mapped over: their axis `axis` is the batch axis,
#' and `f` sees one slice of it at a time, with that axis dropped. Every leaf of a
#' nested argument is mapped over. The other arguments are passed to each
#' application of `f` unchanged. Every output of `f` is stacked along `axis`.
#' @param f (`function`)\cr
#'   Function to vectorize.
#' @param args (`character` | `integer` | `NULL`)\cr
#'   Names or positions of the arguments to map over. If `NULL` (the default),
#'   all of them are mapped over.
#' @param axis (`integer(1)`)\cr
#'   The axis of the arguments in `args` that is mapped over, and the axis of
#'   the outputs along which the results are stacked. All arguments in `args`
#'   must have the same size along it.
#' @return (`function`)\cr
#'   Has the same formals as `f` and must be called inside [`jit()`]. It returns
#'   what `f` returns, with every array gaining the batch axis at `axis`.
#' @section Supported Primitives:
#' A primitive applied to a value that is mapped over needs a `vectorize` rule
#' (see [`rule_vectorize()`]); calling one that has none raises an error.
#' Values that are not mapped over can go through any primitive.
#' @seealso [`rule_vectorize()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' # f is written for a single vector
#' f <- function(x, y) sum(x * y)
#' x <- nv_array(matrix(1:6, nrow = 3), dtype = "f32")
#'
#' # one dot product per row of x, against the same y
#' jit(vectorize(f, args = "x"))(x, nv_array(c(1, 2), dtype = "f32"))
#'
#' # map over the columns instead
#' jit(vectorize(f, args = "x", axis = 2L))(x, nv_array(c(1, 2, 3), dtype = "f32"))
vectorize <- function(f, args = NULL, axis = 1L) {
  assert_function(f)
  mapped_args <- resolve_arg_names(f, args, "args")
  if (!is.null(mapped_args) && !all(mapped_args %in% formalArgs(f))) {
    cli_abort("{.arg args} must be a subset of the formal arguments of {.arg f}.")
  }
  axis <- as.integer(checkmate::assert_int(axis, lower = 1L))
  f_vectorized <- function() {
    call_args <- as.list(match.call())[-1L]
    call_args <- lapply(call_args, eval, envir = parent.frame())
    if (is.null(current_descriptor(silent = TRUE))) {
      cli_abort(c(
        "{.fn vectorize} can only be called inside a {.fn jit}-compiled function.",
        i = "Wrap the result of {.fn vectorize} in {.fn jit}, e.g. {.code jit(vectorize(f))}."
      ))
    }
    vectorize_call(f, call_args, mapped_args, axis)
  }
  formals(f_vectorized) <- formals2(f)
  f_vectorized
}

# Traces `f` on one slice of the arguments `mapped_args` names and replays the
# graph into the current descriptor, batched over `axis`.
vectorize_call <- function(f, call_args, mapped_args, axis) {
  desc <- current_descriptor()
  args_flat <- flatten(call_args)
  in_tree <- build_tree(call_args)
  is_mapped <- if (is.null(mapped_args)) {
    rep(TRUE, length(args_flat))
  } else {
    pjrt::tree_leaf_mask(in_tree, mapped_args)
  }

  size <- NULL
  trace_args <- args_flat
  for (i in which(is_mapped)) {
    x <- args_flat[[i]]
    if (!is_arrayish(x)) {
      cli_abort(c(
        "Can only map over arrays.",
        x = "Got {.cls {class(x)[1L]}}."
      ))
    }
    aval <- to_abstract(x)
    x_shape <- shape(aval)
    if (length(x_shape) < axis) {
      cli_abort(c(
        "Every argument mapped over must have an axis {.val {axis}}.",
        x = "Got an argument of shape {shape_repr(x_shape)}."
      ))
    }
    if (is.null(size)) {
      size <- x_shape[[axis]]
    } else if (x_shape[[axis]] != size) {
      cli_abort(c(
        "Every argument mapped over must have the same size along axis {.val {axis}}.",
        x = "Got sizes {.val {size}} and {.val {x_shape[[axis]]}}."
      ))
    }
    trace_args[[i]] <- if (is_rdata(aval)) {
      RData(shape = x_shape[-axis], r_type = aval$r_type)
    } else {
      AbstractArray(dtype = aval$dtype, shape = x_shape[-axis])
    }
  }
  if (is.null(size)) {
    cli_abort("{.fn vectorize} needs at least one argument to map over.")
  }

  graph <- trace_fn(f, args_flat = trace_args, in_tree = in_tree)
  is_input <- !graph$is_static_flat
  # An R value the body gave a data type materializes at that data type, as for
  # `gradient()` (see `gradient_operands()`).
  operands <- Map(
    function(x, input, mapped) {
      if (is_rdata_box(x)) {
        x <- materialize_rdata(x, input$aval$dtype)
      } else if (is_valid_r(x)) {
        x <- build_r_at(x, input$aval$dtype, desc)
      }
      box <- maybe_box_arrayish(x, desc)
      if (mapped) move_axis(box, axis, 1L) else box
    },
    args_flat[is_input],
    graph$inputs,
    is_mapped[is_input]
  )

  outs <- graph_vectorize(graph, operands, is_mapped[is_input], size)
  unflatten(graph$out_tree, lapply(outs, move_axis, from = 1L, to = axis))
}

# Replays `graph` into the current descriptor as a function of `inputs` (boxes,
# one per graph input), where those that `batched` flags carry a leading batch
# axis of size `size`. A call none of whose inputs is batched is replayed as it
# is; every other call goes through its primitive's vectorize rule, and its
# outputs are batched. Returns the boxes of the outputs, all batched.
graph_vectorize <- function(graph, inputs, batched, size) {
  desc <- current_descriptor()
  boxes <- hashtab()
  is_batched <- hashtab()
  for (i in seq_along(graph$inputs)) {
    boxes[[graph$inputs[[i]]]] <- inputs[[i]]
    is_batched[[graph$inputs[[i]]]] <- batched[[i]]
  }
  const_boxes <- register_consts(desc, graph$constants)
  for (i in seq_along(const_boxes)) {
    boxes[[graph$constants[[i]]]] <- const_boxes[[i]]
  }
  box_of <- function(g) {
    if (is_graph_literal(g)) {
      return(desc$gval_to_box[[g]] %||% GraphBox(g, desc))
    }
    boxes[[g]]
  }
  batched_of <- function(g) !is_graph_literal(g) && isTRUE(is_batched[[g]])

  for (call in graph$statements) {
    in_boxes <- lapply(call$inputs, box_of)
    in_batched <- vapply(call$inputs, batched_of, logical(1L))
    if (!any(in_batched)) {
      desc$statements$add(GraphStatement(call$primitive, lapply(in_boxes, \(b) b$gnode), call$params, call$outputs))
      out_boxes <- register_gvals(desc, call$outputs)
    } else {
      out_boxes <- apply_vectorize_rule(call$primitive, in_boxes, in_batched, call$params, size)
    }
    for (j in seq_along(call$outputs)) {
      boxes[[call$outputs[[j]]]] <- out_boxes[[j]]
      is_batched[[call$outputs[[j]]]] <- any(in_batched)
    }
  }

  lapply(graph$outputs, function(g) batch_operand(box_of(g), batched_of(g), size))
}

apply_vectorize_rule <- function(primitive, inputs, batched, params, size) {
  rule <- primitive[["vectorize"]]
  if (is.null(rule)) {
    cli_abort(c(
      "{.fn vectorize} does not support {.fn {paste0('prim_', primitive$name)}} yet.",
      i = "Only values that are not mapped over can go through it."
    ))
  }
  if (!is.null(rule$fn)) {
    return(rule$fn(inputs, batched, params, size))
  }

  for (i in rule$unbatched) {
    if (batched[[i]]) {
      cli_abort(
        "{.fn vectorize} cannot map over operand {i} of {.fn {paste0('prim_', primitive$name)}}."
      )
    }
  }
  operands <- batch_operands(inputs, batched, size, scalar = rule$scalar, unbatched = rule$unbatched)
  for (nm in names(rule$params)) {
    params[[nm]] <- rule$params[[nm]](params[[nm]], size)
  }
  out <- do.call(primitive_env[[primitive$name]], c(operands, params))
  if (is_graph_box(out)) list(out) else out
}

#' @title Vectorize Rule
#' @description
#' Construct the rule [`vectorize()`] uses for a primitive applied to values
#' that are mapped over. Inside the rule, such values carry the batch axis as
#' their first axis.
#'
#' Most primitives do not need code of their own: they work on an extra
#' leading axis already, and only the parameters that name axes or sizes
#' change. Declaring how each of them changes is enough, via `params`. The
#' rule then gives every operand the batch axis -- broadcasting the ones that
#' are not mapped over -- and applies the primitive with the changed
#' parameters, so all of its outputs are batched. A parameter `params` does not
#' name stays as it is.
#'
#' For a primitive that this does not fit, pass `fn` instead, with the
#' signature `function(inputs, batched, params, size)`: `inputs` are the
#' operands, `batched` says which of them carry the batch axis, and `size` is
#' its size. It returns the outputs, each with the batch axis first.
#' @param fn (`NULL` | `function`)\cr
#'   A rule of its own. If given, the other arguments must be left at their
#'   defaults.
#' @param params (named `list()`)\cr
#'   How each parameter that refers to the axes of the operands changes, one
#'   of the kinds in [`param_axes()`].
#' @param scalar (`integer()`)\cr
#'   Positions of the operands that may be scalars while the others are not,
#'   e.g. the bounds of [`prim_clamp()`]. Such an operand stays a scalar when it
#'   is not mapped over.
#' @param unbatched (`integer()`)\cr
#'   Positions of the operands that cannot be mapped over, e.g. the padding
#'   value of [`prim_pad()`]. They are passed on unchanged.
#' @return (`anvl_rule_vectorize`)
#' @seealso [`vectorize()`], [`param_axes()`]
#' @export
#' @examples
#' # the rule of prim_transpose()
#' rule_vectorize(params = list(perm = param_axis_map()))
rule_vectorize <- function(fn = NULL, params = list(), scalar = integer(), unbatched = integer()) {
  checkmate::assert_function(fn, null.ok = TRUE)
  checkmate::assert_list(params, types = "anvl_param_kind", names = "unique")
  checkmate::assert_integerish(scalar, lower = 1L)
  checkmate::assert_integerish(unbatched, lower = 1L)
  if (!is.null(fn) && (length(params) || length(scalar) || length(unbatched))) {
    cli_abort("Provide either {.arg fn} or {.arg params}, {.arg scalar} and {.arg unbatched}.")
  }
  structure(
    list(fn = fn, params = params, scalar = as.integer(scalar), unbatched = as.integer(unbatched)),
    class = "anvl_rule_vectorize"
  )
}

#' @title Parameter Kinds
#' @description
#' How a parameter of a primitive changes when its operands gain a leading
#' batch axis of size `size`, for [`rule_vectorize()`]:
#'
#' * `param_axes()`: axis indices, which move one axis further (`axes + 1`).
#' * `param_axis_map()`: one axis per axis of an operand, such as a
#'   permutation; the batch axis maps to the batch axis (`c(1, axes + 1)`).
#' * `param_shape()`: a shape, which gains the batch axis (`c(size, shape)`).
#' * `param_per_axis()`: one entry per axis of an operand, with `value` for the
#'   batch axis.
#' @param value (`integer(1)` | `function`)\cr
#'   The entry for the batch axis, or a function of `size` that returns it.
#' @return (`anvl_param_kind`)\cr
#'   A function of the parameter's value and `size` that returns its new value.
#' @export
#' @examples
#' param_axes()(c(1L, 3L), size = 5L)
#' param_axis_map()(c(2L, 1L), size = 5L)
#' param_shape()(c(2L, 3L), size = 5L)
#' param_per_axis(0L)(c(1L, 2L), size = 5L)
param_axes <- function() {
  param_kind(function(x, size) x + 1L)
}

#' @rdname param_axes
#' @export
param_axis_map <- function() {
  param_kind(function(x, size) c(1L, x + 1L))
}

#' @rdname param_axes
#' @export
param_shape <- function() {
  param_kind(function(x, size) c(size, x))
}

#' @rdname param_axes
#' @export
param_per_axis <- function(value) {
  if (!is.function(value)) {
    checkmate::assert_int(value)
  }
  first <- if (is.function(value)) value else function(size) value
  param_kind(function(x, size) c(as.integer(first(size)), x))
}

param_kind <- function(fn) {
  structure(fn, class = "anvl_param_kind")
}

#' @export
print.anvl_rule_vectorize <- function(x, ...) {
  cat("<anvl_rule_vectorize>\n")
  invisible(x)
}

# `x` with a leading batch axis of size `size`: as it is if it already has one,
# broadcast otherwise.
batch_operand <- function(x, batched, size) {
  if (batched) {
    return(x)
  }
  prim_broadcast_in_axes(x, c(size, shape(x)), seq_len(naxes(x)) + 1L)
}

# The operands of a primitive whose operands share one shape, each given the
# batch axis -- except those at the positions `unbatched` names, and a scalar at
# a position `scalar` names when the others are not scalars: such an operand is
# broadcast to the others' shape if it is batched, and left a scalar otherwise.
batch_operands <- function(inputs, batched, size, scalar = integer(), unbatched = integer()) {
  inner_shape <- function(i) {
    s <- shape(inputs[[i]])
    if (batched[[i]]) s[-1L] else s
  }
  main <- setdiff(seq_along(inputs), c(scalar, unbatched))
  full <- if (length(main)) inner_shape(main[[1L]])
  lapply(seq_along(inputs), function(i) {
    x <- inputs[[i]]
    if (i %in% unbatched) {
      return(x)
    }
    if (i %in% scalar && !identical(inner_shape(i), full)) {
      if (!batched[[i]]) {
        return(x)
      }
      return(prim_broadcast_in_axes(x, c(size, full), 1L))
    }
    batch_operand(x, batched[[i]], size)
  })
}

# `x` with its axis `from` moved to position `to`.
move_axis <- function(x, from, to) {
  if (from == to) {
    return(x)
  }
  perm <- seq_len(naxes(x))[-from]
  perm <- append(perm, from, after = to - 1L)
  prim_transpose(x, perm)
}
