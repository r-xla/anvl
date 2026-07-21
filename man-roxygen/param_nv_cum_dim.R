#' @param dim (`integer(1)` | `NULL`)\cr
#'   Dimension along which to accumulate. Negative values count from the end,
#'   i.e. `-1` refers to the last dimension. If `NULL` (default), the input
#'   is first flattened to a 1-D array, like [base::<%= cum_base_fn %>()].
