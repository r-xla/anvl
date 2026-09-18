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
#' @return (`AnvlPrimitive`)
#' @export
AnvlPrimitive <- function(name, subgraphs = character()) {
  checkmate::assert_string(name)
  checkmate::assert_character(subgraphs)

  env <- new.env(parent = emptyenv())
  env$name <- name
  env$rules <- list()
  env$subgraphs <- subgraphs

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
#' Create a new primitive.
#' For details on how to do this, see the article on *Adding a Primitive*.
#' Like every jitted function it runs on the active backend when called.
#' @param name (`character(1)`)\cr
#'   Primitive name.
#' @param fn (`function`)\cr
#'   Body of the primitive. Its formals become the formals of the returned
#'   JIT-compiled callable. The argument is evaluated with the symbol `self`
#'   bound to the new primitive (an [`AnvlPrimitive`]), which is what
#'   [`graph_desc_add()`] takes as its first argument: an inline
#'   `function(...)` body closes over `self`, and a body built by a helper
#'   takes it as an argument (`make_binary_op(self, ...)`).
#' @param subgraphs (`character()`)\cr
#'   Names of parameters that are subgraphs (for higher-order primitives).
#' @param static (`character()` | `integer()`)\cr
#'   Passed to [`jit()`].

#' @param register (`logical(1)`)\cr
#'   If `TRUE` (default), register the result under `name` in the primitive
#'   registry.
#' @return A callable of class `c("JitPrimitive", "JitFunction")`.
#' @export
new_primitive <- function(
  name,
  fn,
  subgraphs = character(),
  static = character(),
  register = TRUE
) {
  checkmate::assert_string(name)
  checkmate::assert_character(subgraphs)
  checkmate::assert_flag(register)

  primitive <- AnvlPrimitive(name, subgraphs = subgraphs)

  # `fn` is evaluated with `self` (the AnvlPrimitive) bound in a per-primitive
  # env around the call site, so the body can reference the primitive directly
  # — same idea as R6's `self`. An inline `function(...)` body closes over that
  # env; a shared body built by one of the `make_*_op()` helpers takes `self`
  # as an argument instead, because a helper's body is written inside another
  # function, where a free `self` would be a global variable.
  self_env <- new.env(parent = parent.frame())
  self_env$self <- primitive
  fn <- eval(substitute(fn), self_env)
  checkmate::assert_function(fn)

  jit_fn <- jit(fn, static = static)
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
