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
#' @section Operands that must agree:
#' A primitive promotes nothing on its own. Where its operands have to agree on
#' a data type, the body says so itself, on the values it goes on to hand
#' [`graph_desc_add()`]:
#'
#' ```r
#' function(lhs, rhs) {
#'   operands <- promote_operands(list(lhs = lhs, rhs = rhs), promote_yield())
#'   graph_desc_add(self, operands, infer_fn = infer_fn)[[1L]]
#' }
#' ```
#'
#' [`promote_yield()`] is the rule for it: an operand that has a data type keeps
#' it, and an R value takes the one the others have, within its own category.
#' That is what makes `prim_mul(x_f64, 2)` work whatever `x`'s data type is,
#' while keeping a primitive from widening the array it was handed -- which is
#' the `nv_*` layer's job.
#'
#' Pass only the operands that must agree. `prim_ifelse()` promotes its two
#' branches and leaves `pred` a `bool`; `prim_scatter()` promotes `x` and
#' `update` and leaves the indices alone. A primitive with one arrayish operand,
#' or with deliberately heterogeneous ones ([`prim_sort()`]'s payload,
#' [`prim_while()`]'s loop state), calls nothing.
#'
#' Put the call before the body uses the operands for anything else, so it sees
#' settled data types throughout: `prim_reduce()` reads `dtype(init)` to trace
#' its reductor and `prim_scatter()` builds its update computation's parameter
#' slots from `peek_dtype(x)`, both before recording a call.
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
  static = character(),
  device = NULL,
  register = TRUE
) {
  checkmate::assert_string(name)
  checkmate::assert_function(fn)
  checkmate::assert_character(subgraphs)
  checkmate::assert_flag(register)

  primitive <- AnvlPrimitive(name, subgraphs = subgraphs)

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
