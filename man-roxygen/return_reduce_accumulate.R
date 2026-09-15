#' @return [`arrayish`]\cr
#'   Has the same data type as the input, except for a boolean input, which is
#'   accumulated at the default integer data type (see [`default_dtypes()`]).
#'   When `drop = TRUE`, the reduced axes are removed.
#'   When `drop = FALSE`, the reduced axes are set to 1.
