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
#' @param promote (`NULL` | [`PromoteRule`][promote_rule] | `list`)\cr
#'   How the primitive brings its arrayish arguments to one data type before it
#'   records a call. See [`new_primitive()`].
#' @return (`AnvlPrimitive`)
#' @export
AnvlPrimitive <- function(name, subgraphs = character(), promote = promote_yield()) {
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
#' @param promote (`NULL` | [`PromoteRule`][promote_rule] | `list`)\cr
#'   How the primitive brings its arrayish arguments to one data type before it
#'   records a call, applied to the `args` of [`graph_desc_add()`].
#'
#'   Defaults to [`promote_yield()`]: all of them must agree, an argument that
#'   has a data type keeps it, and an R value takes the one the others have --
#'   within its own category. Restrict it with `only =` for a primitive whose
#'   operands are *meant* to differ in part, such as [`prim_ifelse()`]'s `pred`
#'   or a gather's indices.
#'
#'   `NULL` means no rule at all: every R value commits to its own default, and
#'   the arguments may hold any data types the primitive accepts. That is what
#'   [`prim_sort()`] and [`prim_while()`] want, since a sort payload and a
#'   loop-carried state are deliberately heterogeneous.
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
  promote = promote_yield(),
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

  jit_fn <- jit(fn, static = static, backend = "auto", device = device)
  attr(jit_fn, "primitive") <- primitive
  class(jit_fn) <- c("JitPrimitive", class(jit_fn))

  if (register) {
    assign(name, jit_fn, envir = primitive_env)
  }

  jit_fn
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
