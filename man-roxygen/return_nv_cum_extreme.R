#' @return ([`arrayish`] | named `list` of two [`arrayish`])\cr
#'   One array when `with_indices = FALSE`, a named `list` of `values` and
#'   `indices` when `with_indices = TRUE`. The values have the input's data
#'   type and the indices `i32`; both have the input's shape when `axis` is
#'   given, and are 1-D of length `prod(shape(x))` when `axis` is `NULL`.
