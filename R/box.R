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
as_array.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as_array")
}

#' @export
as_raw.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as_raw")
}

#' @method as.array GraphBox
#' @export
as.array.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as.array")
}

#' @method as.matrix GraphBox
#' @export
as.matrix.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as.matrix")
}

#' @method as.vector GraphBox
#' @export
as.vector.GraphBox <- function(x, mode = "any") {
  abort_box_to_r(x, "as.vector")
}

#' @method as.list GraphBox
#' @export
as.list.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as.list")
}

#' @method as.double GraphBox
#' @export
as.double.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as.double")
}

#' @method as.integer GraphBox
#' @export
as.integer.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as.integer")
}

#' @method as.logical GraphBox
#' @export
as.logical.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as.logical")
}

#' @method as.raw GraphBox
#' @export
as.raw.GraphBox <- function(x) {
  abort_box_to_r(x, "as.raw")
}

#' @method as.character GraphBox
#' @export
as.character.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "as.character")
}

#' @method as.integer64 GraphBox
#' @exportS3Method bit64::as.integer64
as.integer64.GraphBox <- function(x, ...) {
  abort_box_to_r(x, "bit64::as.integer64")
}
