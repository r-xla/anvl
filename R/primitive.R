#' @title Primitive Definition
#' @description
#' The definition of a primitive: its name, sub-graph parameters and
#' interpretation rules. It is not callable; the function a primitive is
#' called through is an [`AnvlPrimitive`], which carries
#' its `AnvlPrimitiveDef` as `attr(<fn>, "definition")`. The graph records the
#' `AnvlPrimitiveDef` in each [`GraphStatement`].
#' Note that `[[` and `[[<-` access the interpretation rules.
#' To access other fields, use `$` and `$<-`.
#'
#' A primitive is considered higher-order if it has subgraphs.
#' @param name (`character(1)`)\cr
#'   The name of the primitive, without the `prim_` prefix.
#' @param subgraphs (`character()`)\cr
#'   Names of parameters that are subgraphs.
#' @return (`AnvlPrimitiveDef`)
#' @export
AnvlPrimitiveDef <- function(name, subgraphs = character()) {
  checkmate::assert_string(name)
  checkmate::assert_character(subgraphs)

  env <- new.env(parent = emptyenv())
  env$name <- name
  env$rules <- list()
  env$subgraphs <- subgraphs

  structure(env, class = "AnvlPrimitiveDef")
}


primitive_env <- new.env(parent = emptyenv())

# The `AnvlPrimitiveDef` of `x`, which is either that definition or the
# `AnvlPrimitive` callable carrying it.
primitive_def <- function(x) {
  if (inherits(x, "AnvlPrimitive")) attr(x, "definition") else x
}

is_higher_order_primitive <- function(x) {
  length(primitive_def(x)$subgraphs) > 0L
}


#' @method [[<- AnvlPrimitiveDef
#' @export
`[[<-.AnvlPrimitiveDef` <- function(x, name, value) {
  if (!(name %in% globals$interpretation_rules)) {
    cli_abort("Invalid field name {.field {name}} for primitive {.field {x$name}}")
  }
  x$rules[[name]] <- value
  x
}

#' @method [[ AnvlPrimitiveDef
#' @export
`[[.AnvlPrimitiveDef` <- function(x, name) {
  if (name %in% globals$interpretation_rules) {
    return(x$rules[[name]])
  }
  cli_abort("Invalid field name {.field {name}} for primitive {.field {x$name}}")
}

#' @method print AnvlPrimitiveDef
#' @export
print.AnvlPrimitiveDef <- function(x, ...) {
  cat(sprintf("<AnvlPrimitiveDef:%s>\n", x$name))
  invisible(x)
}

#' @method [[ AnvlPrimitive
#' @export
`[[.AnvlPrimitive` <- function(x, name) {
  attr(x, "definition")[[name]]
}

#' @method [[<- AnvlPrimitive
#' @export
`[[<-.AnvlPrimitive` <- function(x, name, value) {
  attr(x, "definition")[[name]] <- value
  x
}

#' @title Create a Primitive
#' @description
#' `new_primitive()` creates a new primitive: an `AnvlPrimitive`, the function
#' the primitive is called through (e.g. `prim_add()`).
#' For details on how to do this, see `r roxy_article("extending_primitive")`.
#' Like every jitted function it runs on the active backend when called.
#' @param name (`character(1)`)\cr
#'   Primitive name, without the `prim_` prefix (`"add"` for `prim_add()`).
#' @param fn (`function`)\cr
#'   Body of the primitive. Its formals become the formals of the returned
#'   JIT-compiled callable. Inside `fn`, the primitive is accessible via
#'   the lexically-bound symbol `self` (an [`AnvlPrimitiveDef`]); pass it as
#'   the first argument to [`graph_desc_add()`].
#' @param subgraphs (`character()`)\cr
#'   Names of parameters that are subgraphs (for higher-order primitives).
#' @param static (`character()` | `integer()`)\cr
#'   Passed to [`jit()`].
#' @param register (`logical(1)`)\cr
#'   Whether to add the primitive to anvl's internal registry of primitives,
#'   under `name`, replacing one registered under the same name. The quickr
#'   backend reads that registry to know which primitives it can lower, so a
#'   primitive created with `register = FALSE` is rejected on quickr even if
#'   it has a `quickr` rule. The other backends only read the rules of the
#'   primitive itself. This does not bind the result to a `prim_<name>`
#'   variable; assign it yourself.
#' @return (`AnvlPrimitive`)\cr
#'   The function the primitive is called through, of class
#'   `c("AnvlPrimitive", "JitFunction")`. Its [`AnvlPrimitiveDef`] is
#'   `attr(<fn>, "definition")`, and `[[` / `[[<-` on it access the rules of
#'   that definition.
#' @aliases AnvlPrimitive
#' @export
new_primitive <- function(
  name,
  fn,
  subgraphs = character(),
  static = character(),
  register = TRUE
) {
  checkmate::assert_string(name)
  checkmate::assert_function(fn)
  checkmate::assert_character(subgraphs)
  checkmate::assert_flag(register)

  definition <- AnvlPrimitiveDef(name, subgraphs = subgraphs)

  # Bind `self` (the AnvlPrimitiveDef) in a per-primitive env wrapped around fn's
  # existing enclosing env, so the body can reference the primitive directly —
  # same idea as R6's `self`. A per-primitive env is needed because inline
  # `function(...)` literals in R/primitives.R all share the package namespace
  # env; binding `self` there would clobber across primitives.
  self_env <- new.env(parent = environment(fn))
  self_env$self <- definition
  environment(fn) <- self_env
  body(fn) <- mark_primitive_body(body(fn))

  jit_fn <- jit(fn, static = static)
  attr(jit_fn, "definition") <- definition
  class(jit_fn) <- c("AnvlPrimitive", class(jit_fn))

  if (register) {
    assign(name, jit_fn, envir = primitive_env)
  }

  jit_fn
}


# Say which primitive is running, so that whatever it refuses reaches the caller
# as coming from the `prim_*()` they wrote. The marker is set first thing in the
# body, before the wrapper's own checks run, so an error from a helper such as
# `resolve_axes()` or from a `cli_abort()` in the body is reported as the
# primitive (`Error in prim_sum()`) rather than as the helper or the anonymous
# function `jit()` wraps. `trace_fn()` does the rewriting of the error's call,
# and `graph_desc_add()` clears the marker once inference has passed.
mark_primitive_body <- function(body) {
  rlang::expr({
    base::assign("INFER_PRIMITIVE", self, envir = utils::getFromNamespace("globals", "anvl"))
    !!body
  })
}


#' @title Get Subgraphs from Higher-Order Primitive
#' @description
#' Extracts the subgraphs from the parameters of a statement that applies a
#' higher-order primitive, such as the branches of [`prim_if()`] or the body
#' of [`prim_while()`].
#'
#' This is not recursive: only the subgraphs held directly by `statement` are
#' returned, not the ones nested in the statements of those subgraphs. Call
#' `subgraphs()` on their statements to descend further.
#' @param statement (`GraphStatement`)\cr
#'   The statement.
#' @return (named `list(AnvlGraph)`)\cr
#'   The subgraphs, named after the parameters that hold them. Empty for a
#'   primitive that is not higher-order.
#' @export
subgraphs <- function(statement) {
  p <- primitive_def(statement$primitive)
  if (!is_higher_order_primitive(p)) {
    return(list())
  }

  stats::setNames(
    lapply(p$subgraphs, \(sg) statement$params[[sg]]),
    p$subgraphs
  )
}
