#' @return ([`arrayish`] | named `list` of two [`arrayish`])\cr
#'   One array when `indices = FALSE`, a named `list` of `values` and
#'   `indices` when `indices = TRUE`. The values have the input's data
#'   type and the indices the default integer data type; both have the input's shape when `axis` is
#'   given, and are 1-D of length `prod(shape(x))` when `axis` is `NULL`.
