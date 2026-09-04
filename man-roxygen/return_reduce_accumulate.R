#' @return [`arrayish`]\cr
#'   Has the same data type as the input, except for a boolean input, which is
#'   accumulated at `i32`: `TRUE` counts as one, rather than being folded with
#'   a logical or/and.
#'   When `drop = TRUE`, the reduced axes are removed.
#'   When `drop = FALSE`, the reduced axes are set to 1.
