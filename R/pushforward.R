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
#' column by column, as [`jacobian()`] does.
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
#'   array. An argument passed through `...` without a name gets its tangent
#'   at the same position. An argument without a tangent is held fixed, and
#'   `NULL` in place of an array -- or of a whole part of an argument -- stands
#'   for zeros. An array that is not a float, and an R value without a data
#'   type such as `2`, has no tangent: pass `NULL` in its place.
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
  trace_pushforward(.f, differentiated_args(.f, list(...)), "pushforward")$pushforward
}

#' @rdname pushforward
#' @export
value_and_pushforward <- function(.f, ...) {
  assert_function(.f)
  trace_pushforward(.f, differentiated_args(.f, list(...)), "value_and_pushforward")[c("value", "pushforward")]
}

# The pushforward of `f` for `pushforward()`, `value_and_pushforward()` or
# `jacobian()`, named by `fn`, at the arguments `prep` holds (see
# `differentiated_args()`): `f` traced into a graph whose forward pass is
# replayed into the enclosing trace, and the function that records the
# tangents of a direction there. `flat` holds the same on the leaves: the boxes
# of the value, its tree `out_tree`, and a pushforward from a tangent per leaf
# `prep$is_wrt_flat` selects to a tangent per leaf of the value (`NULL` for one
# that is not a float).
trace_pushforward <- function(f, prep, fn, call = rlang::caller_env()) {
  traced <- trace_replayed(f, prep, fn, call = call, replay = graph_pushforward)
  graph <- traced$graph
  # the tangents of all leaves of the arguments, `NULL` for one held fixed
  push_leaves <- function(tangents) {
    assert_called_within(traced$enclosing, "pushforward", fn)
    traced$res$pushforward(tangents[!graph$is_static_flat])
  }
  push_flat <- function(tangents_wrt) {
    tangents <- vector("list", length(prep$args_flat))
    tangents[prep$is_wrt_flat] <- tangents_wrt
    push_leaves(tangents)
  }
  push <- function(...) {
    assert_called_within(traced$enclosing, "pushforward", fn)
    tangents <- flatten_tangents(f, prep, list(...), call = rlang::current_env())
    unflatten(graph$out_tree, push_leaves(tangents))
  }
  list(
    value = unflatten(graph$out_tree, traced$res$value),
    pushforward = push,
    flat = list(value = traced$res$value, out_tree = graph$out_tree, pushforward = push_flat)
  )
}

# The tangents `tangents` of the arguments `prep` holds, given as `f` takes
# them, as a list aligned with the leaves of the arguments: a box of the
# current trace for a float array, `NULL` for a leaf held fixed. A tangent goes
# to the argument of its name, or one passed through `...` without a name to
# the argument at the same position among those. Errors name `call`.
flatten_tangents <- function(f, prep, tangents, call) {
  args <- unflatten(prep$in_tree, prep$args_flat)
  tangents <- match_args_to_formals(f, tangents)
  arg_keys <- argument_keys(args)
  tangent_keys <- argument_keys(tangents)
  unknown <- setdiff(tangent_keys, arg_keys)
  if (length(unknown)) {
    cli_abort(
      c(
        "A pushforward takes tangents only of the arguments its function was evaluated at.",
        x = "Got one for {.arg {unknown}}."
      ),
      call = call
    )
  }
  # the argument each leaf belongs to
  leaf_arg <- rep(seq_along(args), pjrt::tree_child_sizes(prep$in_tree))
  leaves <- vector("list", length(prep$args_flat))
  for (k in seq_along(tangents)) {
    j <- match(tangent_keys[[k]], arg_keys)
    mismatch <- sprintf("The tangent of {.arg %s} must be structured like the argument.", tangent_keys[[k]])
    leaves[leaf_arg == j] <- leaves_like(args[[j]], tangents[[k]], call, mismatch, is_leaf = is_rng_state)
  }
  Map(
    function(primal, tangent, is_float, i) {
      if (is.null(tangent)) {
        return(NULL)
      }
      at <- pjrt::tree_path(prep$in_tree, i)
      if (!is_float) {
        what <- if (!is_arrayish(primal)) {
          cli::format_inline("{.cls {class(primal)[1L]}}")
        } else if (has_no_dtype(primal)) {
          "an R value without a data type"
        } else {
          cli::format_inline("an array of data type {.val {as.character(dtype(primal))}}")
        }
        cli_abort(
          c(
            "Can only pass a tangent for a float array.",
            x = "{.code {at}} is {what}.",
            i = "Pass {.code NULL} in its place."
          ),
          call = call
        )
      }
      check_seed(
        primal,
        tangent,
        kind = "tangent",
        where = cli::format_inline(" of {.code {at}}"),
        of = "argument",
        call = call
      )
    },
    prep$args_flat,
    leaves,
    prep$is_wrt_flat,
    seq_along(prep$args_flat)
  )
}

# The key of each of the arguments `args` of a call, matched to the formals of
# its function: its name, or `..i` for the i-th argument passed through `...`
# without one.
argument_keys <- function(args) {
  keys <- names(args) %||% rep("", length(args))
  unnamed <- !nzchar(keys)
  keys[unnamed] <- paste0("..", seq_len(sum(unnamed)))
  keys
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
  rebuilt <- rebuild_forward_into(graph, current_descriptor(), inputs, required_env = hashtab())
  needed <- values_needed(graph, graph$outputs)$kept

  push <- function(tangents) {
    tangent_env <- hashtab()
    for (i in seq_along(graph$inputs)) {
      if (!is.null(tangents[[i]])) {
        tangent_env[[graph$inputs[[i]]]] <- tangents[[i]]
      }
    }
    tangent_of <- function(g) if (!is_graph_literal(g)) tangent_env[[g]]

    for (i in which(needed)) {
      call <- graph$statements[[i]]
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
        lapply(call$inputs, rebuilt$box_of),
        lapply(call$outputs, rebuilt$box_of),
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
      tangent_of(out) %||% zeros(out$aval$dtype, shape(out$aval))
    })
  }
  list(value = lapply(graph$outputs, rebuilt$box_of), pushforward = push)
}
