#' @title Jacobian-Vector Products
#' @description
#' Return a new function that computes the derivative of `f` in a direction of
#' its inputs via forward-mode automatic differentiation: the Jacobian of `f`
#' times the tangent vector. Where [`gradient()`] gives the derivative of a
#' single scalar with respect to all inputs, `jvp()` gives the derivative of
#' all outputs along a single direction of the inputs. The returned function
#' has the arguments of `f` and one more, `tangents`: the direction.
#'
#' * `jvp()` returns only the derivative, structured like the output of `f`.
#' * `value_and_jvp()` returns both the output of `f` and its derivative,
#'   computed in a single pass.
#'
#' Forward mode does not cover every primitive yet: the *Forward* column of
#' `r roxy_article("primitives")` shows which primitives have a forward rule.
#' @param f (`function`)\cr
#'   Function to differentiate. Must not have an argument named `tangents`.
#' @param wrt (`character` | `integer` | `NULL`)\cr
#'   Names or positions of the arguments to differentiate with respect to;
#'   the others are held fixed. As for [`gradient()`], only float arrays can
#'   be included, and at call time an argument in `wrt` must be an array with
#'   a data type, not an R value such as `3`.
#'   If `NULL` (the default), all arguments are included.
#' @return (`function`)\cr
#'   Has the formals of `f` followed by `tangents`, and must be called inside
#'   [`jit()`]. `tangents` is the direction to differentiate in: a named
#'   `list` with one tangent per argument in `wrt` (or of `f`), structured like
#'   the gradients [`gradient()`] returns: an array of the data type and shape
#'   of each array of the argument. An R value takes the data type of its
#'   array.
#'   For `jvp()`, it returns the derivative of the output of `f` in the
#'   direction of `tangents`, structured like that output; an output that is
#'   not a float has a zero derivative.
#'   For `value_and_jvp()`, it returns `list(value = , tangent = )`: the
#'   return value of `f`, and that derivative.
#' @seealso `r roxy_article("autodiff")`
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f <- function(x, y) sin(x) * y
#' x <- nv_array(c(0, 1), dtype = "f64")
#' y <- nv_array(c(2, 3), dtype = "f64")
#' v <- nv_array(c(1, 1), dtype = "f64")
#'
#' # the derivative in the direction of `x`: cos(x) * y
#' g <- jit(jvp(f, wrt = "x"))
#' g(x, y, tangents = list(x = v))
#'
#' # along both arguments, together with the value of `f`
#' vg <- jit(value_and_jvp(f))
#' vg(x, y, tangents = list(x = v, y = v))
#'
#' # the product of the Hessian of a function with a vector
#' hvp <- jit(jvp(gradient(function(x) sum(x * sin(x)))))
#' hvp(x, tangents = list(x = v))
jvp <- function(f, wrt = NULL) {
  assert_function(f)
  wrt <- resolve_jvp_wrt(f, wrt)
  f_jvp <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    trace_value_and_jvp(f, args, wrt, "jvp")$tangent
  }
  formals(f_jvp) <- c(formals2(f), alist(tangents = ))
  f_jvp
}

#' @rdname jvp
#' @export
value_and_jvp <- function(f, wrt = NULL) {
  assert_function(f)
  wrt <- resolve_jvp_wrt(f, wrt)
  f_value_and_jvp <- function() {
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    trace_value_and_jvp(f, args, wrt, "value_and_jvp")
  }
  formals(f_value_and_jvp) <- c(formals2(f), alist(tangents = ))
  f_value_and_jvp
}

# The `wrt` of jvp() / value_and_jvp() as names of `f`'s formals, after
# checking that `f` leaves the name `tangents` to the returned function.
resolve_jvp_wrt <- function(f, wrt, call = rlang::caller_env()) {
  if ("tangents" %in% formalArgs2(f)) {
    cli_abort(
      c(
        "{.arg f} must not have an argument named {.arg tangents}.",
        i = "The returned function takes the direction to differentiate in as {.arg tangents}."
      ),
      call = call
    )
  }
  resolve_wrt(f, wrt, call = call)
}

# jvp() / value_and_jvp() (named by `fn`, which errors name) at the call's
# arguments: `f` traced into a graph whose value and tangent are replayed into
# the enclosing trace.
trace_value_and_jvp <- function(f, args, wrt, fn) {
  call <- call(fn)
  if (!"tangents" %in% names(args)) {
    cli_abort("{.arg tangents} is missing: the direction to differentiate in.", call = call)
  }
  tangents <- args[["tangents"]]
  args[["tangents"]] <- NULL
  prep <- prepare_gradient_args(args, wrt)
  assert_inside_jit(fn, call)
  tangents_flat <- align_tangents(args, wrt, tangents, prep$in_tree, call)
  traced <- trace_and_replay(f, prep, function(graph, operands) {
    graph_value_and_jvp(graph, operands, tangents_flat[!graph$is_static_flat])
  })
  out_tree <- traced$graph$out_tree
  list(
    value = unflatten(out_tree, traced$res$value),
    tangent = unflatten(out_tree, traced$res$tangent)
  )
}

# The tangents as a list aligned with the leaves of `args`, `NULL` for a leaf
# held fixed, each boxed in the current descriptor. Errors name `call`.
align_tangents <- function(args, wrt, tangents, in_tree, call) {
  wrt <- wrt %||% rlang::names2(args)
  if (!all(nzchar(wrt))) {
    cli_abort(
      c(
        "Can only differentiate with respect to arguments with a name.",
        i = "Pass the arguments by name, or select them with {.arg wrt}."
      ),
      call = call
    )
  }
  is_list <- is.list(tangents) && !is.object(tangents)
  if (!is_list || !setequal(rlang::names2(tangents), wrt) || anyDuplicated(names(tangents))) {
    cli_abort(
      c(
        "{.arg tangents} must be a list with one tangent per argument differentiated with respect to, named by it.",
        i = "Expected the names {.arg {wrt}}.",
        x = if (is_list) {
          "Got {.arg {rlang::names2(tangents)}}."
        } else if (is_arrayish(tangents)) {
          "Got a single array."
        } else {
          "Got {.cls {class(tangents)[[1L]]}}."
        }
      ),
      call = call
    )
  }
  out <- vector("list", tree_size(in_tree))
  for (nm in wrt) {
    if (build_tree(tangents[[nm]], is_leaf = is_rng_state) != build_tree(args[[nm]], is_leaf = is_rng_state)) {
      cli_abort("The tangent of {.arg {nm}} must be structured like the argument.", call = call)
    }
    out[pjrt::tree_leaf_mask(in_tree, nm)] <- Map(
      check_tangent,
      flatten(args[[nm]], is_leaf = is_rng_state),
      flatten(tangents[[nm]], is_leaf = is_rng_state),
      nm,
      list(call)
    )
  }
  out
}

# The tangent `tangent` of the float array `primal` of argument `arg`, boxed
# in the current descriptor, after checking that the two fit together. Errors
# name `call`.
check_tangent <- function(primal, tangent, arg, call) {
  dt <- dtype(primal)
  if (has_no_dtype(tangent)) {
    tangent <- materialize_at(tangent, dt)
  } else if (!is_arrayish(tangent)) {
    cli_abort(
      c(
        "The tangent of {.arg {arg}} must be an array.",
        x = "Got {.cls {class(tangent)}}."
      ),
      call = call
    )
  }
  tangent <- maybe_box_arrayish(tangent)
  if (dtype(tangent) != dt || !identical(shape(tangent), shape(primal))) {
    cli_abort(
      c(
        "The tangent of {.arg {arg}} must have the data type and shape of the argument.",
        x = "The argument is {.val {format_aval_short(to_abstract(primal))}}, the tangent {.val {format_aval_short(to_abstract(tangent))}}." # nolint
      ),
      call = call
    )
  }
  tangent
}

# Replays `graph` into the current descriptor as a function of the boxes
# `inputs`, as `graph_value_and_grad()` does, and alongside it the tangent of
# every value that a tangent in `tangents` -- one per input, `NULL` for an
# input held fixed -- reaches, by the forward rules of the statements. A value
# no tangent reaches has none: a zero no statement computes. So does a value
# that is not a float, which has no derivative, and a value no output depends
# on, whose tangent nobody reads.
graph_value_and_jvp <- function(graph, inputs, tangents) {
  desc <- current_descriptor()
  rebuilt <- rebuild_forward_into(graph, desc, inputs, required_env = hashtab())
  box_of <- function(g) register_gval(desc, rebuilt$translate(g))

  tangent_env <- hashtab()
  for (i in seq_along(graph$inputs)) {
    if (!is.null(tangents[[i]])) {
      tangent_env[[graph$inputs[[i]]]] <- tangents[[i]]
    }
  }
  tangent_of <- function(g) if (!is_graph_literal(g)) tangent_env[[g]]

  needed <- output_dependencies(graph)
  for (call in graph$statements) {
    if (!any(vapply(call$outputs, \(g) isTRUE(needed[[g]]), logical(1L)))) {
      next
    }
    in_tangents <- lapply(call$inputs, tangent_of)
    if (all(vapply(in_tangents, is.null, logical(1L)))) {
      next
    }
    rule <- call$primitive[["forward"]]
    if (is.null(rule)) {
      cli_abort(c(
        "No forward rule for primitive {.field {call$primitive$name}}.",
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

  list(
    value = lapply(graph$outputs, box_of),
    tangent = lapply(graph$outputs, \(out) tangent_of(out) %||% zeros(dtype(out), shape(out)))
  )
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
