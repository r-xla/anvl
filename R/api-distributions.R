## Probability distributions ---------------------------------------------------

#' @title Normal Density
#' @description
#' Computes the probability density function of the normal distribution:
#' \deqn{f(x) = \frac{1}{\sigma\sqrt{2\pi}}
#'   \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}
#' where \eqn{\mu} is the mean and \eqn{\sigma} is the standard deviation of the
#' distribution.
#' Converts `mean` and `sd` to the data type of `x`.
#' @param x ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density.
#' @param mean ([`arrayish`])\cr
#'   Mean of the distribution (scalar or same shape as `x`).
#' @param sd ([`arrayish`])\cr
#'   Standard deviation of the distribution (scalar or same shape as `x`).
#'   Must be positive, otherwise results are invalid.
#' @param log (`logical(1)`)\cr
#'   If `TRUE`, returns the log-density instead of the density.
#' @template return_unary
#' @seealso [nv_rnorm()] for sampling from a normal distribution.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' nv_dnorm(x)
#' nv_dnorm(x, mean = 1, sd = 2)
#' nv_dnorm(x, log = TRUE)
#' @export
#' @jit static "log"
nv_dnorm <- function(x, mean = 0, sd = 1, log = FALSE) {
  assert_flag(log)
  args <- as_anvl_arrays(x, mean, sd)
  x <- args[[1L]]
  mean <- args[[2L]]
  sd <- args[[3L]]
  op_dtype <- dtype(x)
  mean <- nv_convert(mean, op_dtype)
  sd <- nv_convert(sd, op_dtype)

  z <- (x - mean) / sd
  log_density <- -0.5 * (z * z) - nv_log(sd) - 0.5 * base::log(2 * pi)

  if (log) {
    return(log_density)
  }
  nv_exp(log_density)
}
