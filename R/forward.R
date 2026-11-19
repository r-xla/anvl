#' @title Pushforwards
#' @description
#' Evaluate `.f` at the arguments `...` and return its pushforward: the
#' function that takes a tangent of the arguments -- a direction to move them
#' in -- to the corresponding tangent of the result, by forward-mode automatic
#' differentiation. That is the Jacobian of `.f` times the tangent, the
#' Jacobian-vector product. Where [`pullback()`] gives the derivative of a
#' weighted sum of the result with respect to all arguments, a pushforward
#' gives the derivative of the whole result along a single direction of the
#' arguments.
#'
#' * `pushforward()` returns only the pushforward.
#' * `value_and_pushforward()` returns both the result of `.f` and its
#'   pushforward.
#'
#' `.f` runs once, when `pushforward()` is called. The pushforward can be
#' called more than once, e.g. with every unit vector to build a Jacobian
#' column by column.
#'
#' Forward mode does not cover every primitive yet: the *Pushforward* column of
#' `r roxy_article("primitives")` shows which primitives have a pushforward
#' rule.
#' @inheritParams pullback
#' @return
#'   For `pushforward()`, the pushforward (`function`); for
#'   `value_and_pushforward()`, `list(value = , pushforward = )`: the return
#'   value of `.f`, and its pushforward.
#'
#'   The pushforward takes the tangents of the arguments as `.f` takes the
#'   arguments, matched to its formals as in a call of `.f`. Each is
#'   structured like its argument, with an array of the same shape and data
#'   type for each float array of it; an R value takes the data type of its
#'   array. An argument without a tangent is held fixed, and `NULL` in place of
#'   an array -- or of a whole part of an argument -- stands for zeros. An
#'   array that is not a float has no tangent: pass `NULL` in its place.
#'
#'   The pushforward returns the tangent of the value of `.f`, structured like
#'   it. An array of the value that is not a float has no tangent, and gets
#'   `NULL` in its place.
#'
#'   Like [`gradient()`], `pushforward()` must be called inside [`jit()`]. The
#'   pushforward must be called in the same function, or in a function traced
#'   inside it, such as a branch of [`nv_if()`].
#' @seealso `r roxy_article("autodiff")`, [`pullback()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f <- function(x, y) sin(x) * y
#' x <- nv_array(c(0, 1), dtype = "f64")
#' y <- nv_array(c(2, 3), dtype = "f64")
#' v <- nv_array(c(1, 1), dtype = "f64")
#' # the derivative in the direction `v` of `x`: cos(x) * y * v
#' g <- jit(function(x, y, v) pushforward(f, x, y)(v))
#' g(x, y, v)
#'
#' # along both arguments, together with the value of `f`
#' h <- jit(function(x, y, v) {
#'   out <- value_and_pushforward(f, x, y)
#'   list(value = out$value, tangent = out$pushforward(v, v))
#' })
#' h(x, y, v)
#'
#' # the product of the Hessian of a function with a vector: the pushforward
#' # of its gradient
#' grad <- function(x) gradient(function(x) sum(x * sin(x)))(x)$x
#' hvp <- jit(function(x, v) pushforward(grad, x)(v))
#' hvp(x, v)
pushforward <- function(.f, ...) {
  assert_function(.f)
  trace_pushforward(.f, pullback_args(.f, list(...)), "pushforward")$pushforward
}

#' @rdname pushforward
#' @export
value_and_pushforward <- function(.f, ...) {
  assert_function(.f)
  trace_pushforward(.f, pullback_args(.f, list(...)), "value_and_pushforward")
}

# The pushforward of `f` for `pushforward()` or `value_and_pushforward()`,
# named by `fn`, at the arguments `prep` holds (see `pullback_args()`): `f`
# traced into a graph whose forward pass is replayed into the enclosing trace,
# and the function that records the tangents of a direction there.
trace_pushforward <- function(f, prep, fn, call = rlang::caller_env()) {
  traced <- trace_replayed(f, prep, fn, call = call, replay = graph_pushforward)
  graph <- traced$graph
  push <- function(...) {
    assert_called_within(traced$enclosing, "pushforward", fn)
    tangents <- flatten_tangents(f, prep, list(...), call = rlang::current_env())
    unflatten(graph$out_tree, traced$res$pushforward(tangents[!graph$is_static_flat]))
  }
  list(value = unflatten(graph$out_tree, traced$res$value), pushforward = push)
}

# The tangents `tangents` of the arguments `prep` holds, given as `f` takes
# them, as a list aligned with the leaves of the arguments: a box of the
# current trace for a float array, `NULL` for a leaf held fixed. Errors name
# `call`.
flatten_tangents <- function(f, prep, tangents, call) {
  args <- unflatten(prep$in_tree, prep$args_flat)
  tangents <- match_args_to_formals(f, tangents)
  unknown <- setdiff(names(tangents), names(args))
  if (length(unknown)) {
    cli_abort(
      c(
        "A pushforward takes tangents only of the arguments its function was evaluated at.",
        x = "Got one for {.arg {unknown}}."
      ),
      call = call
    )
  }
  leaves <- vector("list", length(prep$args_flat))
  for (nm in names(tangents)) {
    mismatch <- sprintf("The tangent of {.arg %s} must be structured like the argument.", nm)
    leaves[pjrt::tree_leaf_mask(prep$in_tree, nm)] <- cotangent_leaves(args[[nm]], tangents[[nm]], call, mismatch)
  }
  Map(
    function(primal, tangent, is_float, i) {
      if (is.null(tangent)) {
        return(NULL)
      }
      at <- pjrt::tree_path(prep$in_tree, i)
      if (!is_float) {
        cli_abort(
          c(
            "Can only pass a tangent for a float array.",
            x = "{.code {at}} is not one.",
            i = "Pass {.code NULL} in its place."
          ),
          call = call
        )
      }
      check_tangent(primal, tangent, at, call)
    },
    prep$args_flat,
    leaves,
    prep$is_wrt_flat,
    seq_along(prep$args_flat)
  )
}

# The tangent `tangent` of the float array `primal` at `at`, boxed in the
# current trace, after checking that the two fit together. Errors name `call`.
check_tangent <- function(primal, tangent, at, call) {
  dt <- dtype(primal)
  if (has_no_dtype(tangent)) {
    tangent <- materialize_at(tangent, dt)
  } else if (!is_arrayish(tangent)) {
    cli_abort(
      c(
        "The tangent of {.code {at}} must be an array.",
        x = "Got {.cls {class(tangent)}}."
      ),
      call = call
    )
  }
  tangent <- maybe_box_arrayish(tangent)
  if (dtype(tangent) != dt || !identical(shape(tangent), shape(primal))) {
    cli_abort(
      c(
        "The tangent of {.code {at}} must have the data type and shape of the argument.",
        x = "The argument is {.val {format_aval_short(to_abstract(primal))}}, the tangent {.val {format_aval_short(to_abstract(tangent))}}." # nolint
      ),
      call = call
    )
  }
  tangent
}

# Replays the forward pass of `graph` into the current descriptor, as a
# function of `inputs`, the boxes its inputs are bound to. Returns the boxes of
# its outputs as `value`, and its `pushforward`: a function that takes a
# tangent per input, `NULL` for an input held fixed, records into the
# descriptor current when it is called the tangent of every value a tangent
# reaches, by the pushforward rules of the statements, and returns those of
# the outputs. A value no tangent reaches has none: a zero no statement
# computes. So does a value that is not a float, which has no derivative, and
# a value no output depends on, whose tangent nobody reads.
graph_pushforward <- function(graph, inputs) {
  desc <- current_descriptor()
  rebuilt <- rebuild_forward_into(graph, desc, inputs, required_env = hashtab())
  box_of <- function(g) register_gval(desc, rebuilt$translate(g))
  needed <- output_dependencies(graph)

  push <- function(tangents) {
    tangent_env <- hashtab()
    for (i in seq_along(graph$inputs)) {
      if (!is.null(tangents[[i]])) {
        tangent_env[[graph$inputs[[i]]]] <- tangents[[i]]
      }
    }
    tangent_of <- function(g) if (!is_graph_literal(g)) tangent_env[[g]]

    for (call in graph$statements) {
      if (!any(vapply(call$outputs, \(g) isTRUE(needed[[g]]), logical(1L)))) {
        next
      }
      in_tangents <- lapply(call$inputs, tangent_of)
      if (all(vapply(in_tangents, is.null, logical(1L)))) {
        next
      }
      rule <- call$primitive[["pushforward"]]
      if (is.null(rule)) {
        cli_abort(c(
          "No pushforward rule for primitive {.field {call$primitive$name}}.",
          i = "Cannot compute a tangent through this primitive."
        ))
      }
      out_tangents <- rule(
        lapply(call$inputs, box_of),
        lapply(call$outputs, box_of),
        in_tangents,
        call$params
      )
      for (j in seq_along(call$outputs)) {
        out <- call$outputs[[j]]
        if (!is.null(out_tangents[[j]]) && is_dtype_float(out$aval$dtype)) {
          tangent_env[[out]] <- out_tangents[[j]]
        }
      }
    }

    lapply(graph$outputs, function(out) {
      if (!is_dtype_float(out$aval$dtype)) {
        return(NULL)
      }
      # A 16-bit float zero is a broadcast constant: stablehlo formats no
      # 16-bit float literal for `prim_fill()`.
      tangent_of(out) %||%
        {
          zero <- get_box_or_register_const(current_descriptor(), nv_scalar(0L, dtype = out$aval$dtype))
          nv_broadcast_to(zero, shape(out$aval))
        }
    })
  }
  list(value = lapply(graph$outputs, box_of), pushforward = push)
}

# hashtab(gval -> TRUE) of the values `graph`'s outputs depend on.
output_dependencies <- function(graph) {
  needed <- hashtab()
  for (out in graph$outputs) {
    if (!is_graph_literal(out)) {
      needed[[out]] <- TRUE
    }
  }
  for (call in rev(graph$statements)) {
    if (any(vapply(call$outputs, \(g) isTRUE(needed[[g]]), logical(1L)))) {
      for (g in call$inputs) {
        if (!is_graph_literal(g)) needed[[g]] <- TRUE
      }
    }
  }
  needed
}
