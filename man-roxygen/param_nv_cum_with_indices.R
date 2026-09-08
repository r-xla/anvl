#' @param with_indices (`logical(1)`)\cr
#'   If `FALSE` (default), returns the running-<%= cum_extreme_name %> array.
#'   If `TRUE`, returns `list(values = ..., indices = ...)` where `indices`
#'   is the index of the last occurrence of the running
#'   <%= cum_extreme_name %> at each position, of the default integer data
#'   type (see [`default_dtypes()`]).
#'   When `axis = NULL`, indices refer to the flattened input.
