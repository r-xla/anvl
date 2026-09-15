#' @title AnvlBox
#' @description
#' Virtual S3 base class for [`GraphBox`].
#' @seealso [GraphBox]
#' @name AnvlBox
NULL

is_box <- function(x) {
  inherits(x, "AnvlBox")
}

# Coercion to R -----------------------------------------------------------

# A box stands for a value the compiled program has not computed yet, so there
# is nothing to hand back to R. Without these methods the base generics fall
# through to the underlying list: `as.vector()` and `as.list()` return the box
# itself, `as.character()` returns "<environment>", and the rest abort with
# base R's complaints about lists and dims -- all a long way from the actual
# mistake, which is reading values during tracing.
abort_box_to_r <- function(x, fn) {
  cli_abort(
    c(
      "{.fn {fn}} is not defined for a {.cls {class(x)[1L]}}.",
      x = "A traced array has no values: it stands for the shape and data type of something the compiled program only computes when it is run.",
      i = "Read the values outside {.fn jit}, or use {.fn nv_print} to print an array from inside a traced function.",
      i = "To branch on values, use {.fn nv_select} or {.fn nv_while} instead of R's {.code if} and {.code while}."
    ),
    call = NULL
  )
}

#' @export
as_array.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as_array")
}

#' @export
as_raw.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as_raw")
}

#' @method as.array AnvlBox
#' @export
as.array.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as.array")
}

#' @method as.matrix AnvlBox
#' @export
as.matrix.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as.matrix")
}

#' @method as.vector AnvlBox
#' @export
as.vector.AnvlBox <- function(x, mode = "any") {
  abort_box_to_r(x, "as.vector")
}

#' @method as.list AnvlBox
#' @export
as.list.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as.list")
}

#' @method as.double AnvlBox
#' @export
as.double.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as.double")
}

#' @method as.integer AnvlBox
#' @export
as.integer.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as.integer")
}

#' @method as.logical AnvlBox
#' @export
as.logical.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as.logical")
}

#' @method as.character AnvlBox
#' @export
as.character.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "as.character")
}

#' @method as.integer64 AnvlBox
#' @exportS3Method bit64::as.integer64
as.integer64.AnvlBox <- function(x, ...) {
  abort_box_to_r(x, "bit64::as.integer64")
}
