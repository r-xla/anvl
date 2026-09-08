#' @return ([`arrayish`])\cr
#'   Has the input's shape when `axis` is given, and is 1-D of length
#'   `prod(shape(x))` when `axis` is `NULL`, which flattens first. Has the
#'   input's data type, except for a
#'   boolean input, which is accumulated at `i32`.
