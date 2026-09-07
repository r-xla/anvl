#' @return [`arrayish`]\cr
#'   Has the same data type as the input, except for a boolean input, which is
#'   accumulated at `i32`.
#'   When `drop = TRUE`, the reduced axes are removed.
#'   When `drop = FALSE`, the reduced axes are set to 1.
