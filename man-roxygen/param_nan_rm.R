#' @param nan_rm (`logical(1)`)\cr
#'   How to handle `NaN` values in floating-point inputs. If `FALSE`
#'   (default), `NaN` propagates -- any slice containing `NaN` reduces
#'   to `NaN`. If `TRUE`, `NaN` values are skipped from the reduction;
#'   a slice that is entirely `NaN` reduces to the identity element of
#'   the operation (e.g. `0` for sum, `-Inf` for max), or to `NaN` when
#'   the operation is not well-defined on an empty input (e.g. mean,
#'   variance, quantile). Has no effect on integer or boolean inputs.
