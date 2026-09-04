#' @return [`arrayish`]\cr
#'   Has the same shape as the input, and the same data type, except for a
#'   boolean input, which is accumulated at `i32`: `TRUE` counts as one, rather
#'   than being folded with a logical or/and.
