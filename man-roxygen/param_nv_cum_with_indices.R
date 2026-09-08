#' @param with_indices (`logical(1)`)\cr
#'   If `FALSE` (default), returns the running-<%= cum_extreme_name %> array.
#'   If `TRUE`, returns `list(values = ..., indices = ...)` where `indices`
#'   is the 1-based index of the last occurrence of the running
#'   <%= cum_extreme_name %> at each position, at the [default integer data type][default_dtypes].
#'   When `axis = NULL`, indices refer to the flattened input.
