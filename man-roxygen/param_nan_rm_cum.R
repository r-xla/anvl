#' @param nan_rm (`logical(1)`)\cr
#'   How to handle `NaN` values in floating-point inputs. If `FALSE`
#'   (default), `NaN` propagates forward from its first occurrence. If
#'   `TRUE`, `NaN` is treated as the identity element of the cumulative
#'   op (`0` for sum, `1` for prod, `-Inf` / `+Inf` for max / min) and
#'   contributes nothing to the running value.
