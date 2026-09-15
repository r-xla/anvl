#' @title AnvlBox
#' @description
#' Virtual S3 base class for [`GraphBox`].
#' @seealso [GraphBox]
#' @name AnvlBox
NULL

is_box <- function(x) {
  inherits(x, "AnvlBox")
}

abort_box_to_r <- function(x, fn) {
  cli_abort(
    c(
      "{.fn {fn}} is not defined for a {.cls {class(x)[1L]}}.",
      x = "A traced array has no values: it stands for the shape and data type of something the compiled program only computes when it is run.",
      i = "You can only convert AnvlArrays to R objects outside of jit()."
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
