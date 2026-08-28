#' @title AnvlPrimitive
#' @description
#' Primitive interpretation rule.
#' Note that `[[` and `[[<-` access the interpretation rules.
#' To access other fields, use `$` and `$<-`.
#'
#' A primitive is considered higher-order if it contains subgraphs.
#' @param name (`character()`)\cr
#'   The name of the primitive.
#' @param subgraphs (`character()`)\cr
#'   Names of parameters that are subgraphs. Only used if `higher_order = TRUE`.
#' @param promote (`NULL` | [`PromoteRule`][promote_rule])\cr
#'   How the primitive brings its arrayish arguments to one data type before it
#'   records a call. See [`new_primitive()`].
#' @return (`AnvlPrimitive`)
#' @export
AnvlPrimitive <- function(name, subgraphs = character(), promote = NULL) {
  checkmate::assert_string(name)
  checkmate::assert_character(subgraphs)

  env <- new.env(parent = emptyenv())
  env$name <- name
  env$rules <- list()
  env$subgraphs <- subgraphs
  env$promote <- promote

  structure(env, class = "AnvlPrimitive")
}


primitive_env <- new.env(parent = emptyenv())

is_higher_order_primitive <- function(x) {
  if (inherits(x, "JitPrimitive")) {
    x <- attr(x, "primitive")
  }
  length(x$subgraphs) > 0L
}


#' @method [[<- AnvlPrimitive
#' @export
`[[<-.AnvlPrimitive` <- function(x, name, value) {
  if (name %in% globals$interpretation_rules) {
    x$rules[[name]] <- value
  } else {
    cli_abort("Invalid field name {.field {name}} for primitive {.field {x$name}}")
  }
  x
}

#' @method [[ AnvlPrimitive
#' @export
`[[.AnvlPrimitive` <- function(x, name) {
  if (name %in% globals$interpretation_rules) {
    return(x$rules[[name]])
  }
  cli_abort("Invalid field name {.field {name}} for primitive {.field {x$name}}")
}

#' @method print AnvlPrimitive
#' @export
print.AnvlPrimitive <- function(x, ...) {
  cat(sprintf("<AnvlPrimitive:%s>\n", x$name))
  invisible(x)
}

#' @method [[ JitPrimitive
#' @export
`[[.JitPrimitive` <- function(x, name) {
  attr(x, "primitive")[[name]]
}

#' @method [[<- JitPrimitive
#' @export
`[[<-.JitPrimitive` <- function(x, name, value) {
  attr(x, "primitive")[[name]] <- value
  x
}

#' @title Create a Primitive
#' @description
#' Builds an [`AnvlPrimitive`] metadata object, wraps `fn` with [`jit()`],
#' attaches the metadata via `attr(., "primitive")`, prepends class
#' `"JitPrimitive"`, and (by default) registers the result under `name` in
#' the primitive registry.
#'
#' The backend is always `"auto"` and cannot be configured.
#' @param name (`character(1)`)\cr
#'   Primitive name.
#' @param fn (`function`)\cr
#'   Body of the primitive. Its formals become the formals of the returned
#'   JIT-compiled callable. Inside `fn`, the primitive is accessible via
#'   the lexically-bound symbol `self` (an [`AnvlPrimitive`]); pass it as
#'   the first argument to [`graph_desc_add()`].
#' @param subgraphs (`character()`)\cr
#'   Names of parameters that are subgraphs (for higher-order primitives).
#' @param promote (`NULL` | [`PromoteRule`][promote_rule])\cr
#'   How the primitive brings its arrayish arguments to one data type. The rule
#'   is applied *before the body runs*, to the non-`static` formals in formal
#'   order with `...` spliced in where it sits -- the same values the body goes
#'   on to hand [`graph_desc_add()`] -- so the body sees settled data types from
#'   its first line. That is what a primitive needs when it reads a data type or
#'   traces a sub-graph before it records its call.
#'
#'   `NULL` (default) means no rule: every R value commits to its own default,
#'   and the arguments may hold any data types the primitive accepts. That is
#'   right for a primitive with one arrayish argument, and for one whose
#'   operands are deliberately heterogeneous ([`prim_sort()`]'s payload,
#'   [`prim_while()`]'s loop state).
#'
#'   A primitive whose arrayish arguments must *agree* says so with
#'   [`promote_yield()`]: an argument that has a data type keeps it, and an R
#'   value takes the one the others have -- within its own category. This is
#'   what makes `prim_mul(x_f64, 2)` work whatever `x`'s data type is. Restrict
#'   it with `only =` where some of the operands are meant to differ, such as
#'   [`prim_ifelse()`]'s `pred` or a gather's indices.
#' @param static (`character()` | `integer()`)\cr
#'   Passed to [`jit()`].
#' @param device (`NULL` | `character(1)` | `device_arg()`)\cr
#'   Passed to [`jit()`]. Useful for primitives with no array inputs
#'   (e.g. `prim_fill`) where the device must come from an explicit argument.
#' @param register (`logical(1)`)\cr
#'   If `TRUE` (default), register the result under `name` in the primitive
#'   registry.
#' @return A callable of class `c("JitPrimitive", "JitFunction")`.
#' @export
new_primitive <- function(
  name,
  fn,
  subgraphs = character(),
  promote = NULL,
  static = character(),
  device = NULL,
  register = TRUE
) {
  checkmate::assert_string(name)
  checkmate::assert_function(fn)
  checkmate::assert_character(subgraphs)
  checkmate::assert_flag(register)

  primitive <- AnvlPrimitive(name, subgraphs = subgraphs, promote = promote)

  # Bind `self` (the AnvlPrimitive) in a per-primitive env wrapped around fn's
  # existing enclosing env, so the body can reference the primitive directly —
  # same idea as R6's `self`. A per-primitive env is needed because inline
  # `function(...)` literals in R/primitives.R all share the package namespace
  # env; binding `self` there would clobber across primitives.
  self_env <- new.env(parent = environment(fn))
  self_env$self <- primitive
  environment(fn) <- self_env

  if (!is.null(promote)) {
    fn <- wrap_promote_resolution(fn, primitive, static)
  }

  jit_fn <- jit(fn, static = static, backend = "auto", device = device)
  attr(jit_fn, "primitive") <- primitive
  class(jit_fn) <- c("JitPrimitive", class(jit_fn))

  if (register) {
    assign(name, jit_fn, envir = primitive_env)
  }

  jit_fn
}

# Wrap a primitive body so its arrayish arguments are resolved *before* the body
# runs, rather than when the body records its call. The body then sees settled
# data types throughout, which is what a primitive that uses an operand before
# recording needs: `prim_reduce()` reads `dtype(init)` to trace its reductor,
# and `prim_scatter()` builds its update computation's parameter slots from
# `peek_dtype(x)`. Both would otherwise see an R value that has not taken the
# other operand's data type yet.
#
# The arrayish arguments are the non-`static` formals in formal order, with
# `...` spliced in where it sits -- the same list, in the same order and under
# the same names, that the body goes on to hand `graph_desc_add()`, so a rule's
# `only =` and `promote_like()` name the same things in both places.
#
# Installed only for a primitive that declares a rule; one with `promote = NULL`
# is called directly. This runs on every traced and eager primitive call, so the
# wrapper is specialized here rather than branching per call: where the formals
# hold no `...`, the resolved values are written back over them and the original
# body is inlined, which costs one `mget` and one `list2env` and no second call
# frame.
wrap_promote_resolution <- function(fn, primitive, static) {
  fmls <- formals(fn)
  nms <- names(fmls)
  static_nms <- if (is.character(static)) static else nms[static]
  arrayish <- setdiff(nms, c(static_nms, "..."))
  dots_at <- match("...", nms, nomatch = 0L)
  optional <- arrayish[vapply(fmls[arrayish], function(d) !identical(d, quote(expr = )), logical(1L))]
  if (length(optional)) {
    cli_abort(c(
      "A primitive's arrayish arguments must all be required.",
      x = "{.val {optional}} {?has/have} a default, so the resolver cannot tell an omitted one from a supplied one.", # nolint
      i = "Give it no default, or declare it {.arg static}."
    ))
  }

  env <- new.env(parent = environment(fn))
  env$.primitive <- primitive
  env$.arrayish <- arrayish

  wrapper <- fn
  environment(wrapper) <- env
  if (dots_at) {
    # `...` cannot be written back over, so the body is re-invoked with the
    # resolved values instead of being inlined.
    env$.fn <- fn
    # Where the dots sit among the arrayish arguments, so they are spliced into
    # the position the body would have built them into.
    env$.dots_after <- sum(match(arrayish, nms) < dots_at)
    env$.static_nms <- static_nms
    body(wrapper) <- quote({
      .args <- append(mget(.arrayish), list(...), after = .dots_after)
      do.call(.fn, c(resolve_primitive_args(.primitive, .args), mget(.static_nms)), quote = TRUE)
    })
    return(wrapper)
  }
  prologue <- quote(list2env(resolve_primitive_args(.primitive, mget(.arrayish)), environment()))
  body(wrapper) <- as.call(c(as.name("{"), prologue, body_statements(fn)))
  wrapper
}

# The statements of a function's body, whether or not it is a `{` block.
body_statements <- function(fn) {
  b <- body(fn)
  if (is.call(b) && identical(b[[1L]], as.name("{"))) {
    return(as.list(b)[-1L])
  }
  list(b)
}

#' @title Get Subgraphs from Higher-Order Primitive
#' @description
#' Extracts subgraphs from the parameters of a higher-order primitive call.
#' @param call (`PrimitiveCall`)\cr
#'   The primitive call.
#' @return (`list(AnvlGraph)`)\cr
#'   List of subgraphs found in the parameters.
#' @export
subgraphs <- function(call) {
  p <- call$primitive
  if (inherits(p, "JitPrimitive")) {
    p <- attr(p, "primitive")
  }
  if (!is_higher_order_primitive(p)) {
    return(list())
  }

  stats::setNames(
    lapply(p$subgraphs, \(sg) call$params[[sg]]),
    p$subgraphs
  )
}
