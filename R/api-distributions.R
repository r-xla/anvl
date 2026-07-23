## Probability distributions ---------------------------------------------------

#' @title The Normal Distribution
#' @name nv_normal
#' @description
#' Density (`nv_dnorm`) and distribution function (`nv_pnorm`) of the Normal
#' distribution with mean `mean` and standard deviation `sd`.
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`).
#' @param mean ([`arrayish`])\cr
#'   Mean of the distribution (scalar or same shape as `x`/`q`).
#' @param sd ([`arrayish`])\cr
#'   Standard deviation of the distribution (scalar or same shape as
#'   `x`/`q`). Must be positive, otherwise results are invalid.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the densities/probabilities are given as logarithms.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le q)}; otherwise,
#'   \eqn{P(X > q)}.
#' @details
#' The Normal distribution has probability density function:
#' \deqn{f(x) = \frac{1}{\sigma\sqrt{2\pi}}
#'   \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}
#' where \eqn{\mu} is the mean and \eqn{\sigma} is the standard deviation.
#' The `mean` and `sd` are converted to the data type of `x`/`q`.
#'
#' `nv_pnorm` uses the asymptotic expansion from
#' `r xlamisc::cite_bib("abramowitz1964handbook")`, equation 26.2.12, in the
#' left tail when `log_p = TRUE` to maintain accuracy.
#' @references
#' `r xlamisc::format_bib("abramowitz1964handbook")`
#' @template return_unary
#' @seealso [nv_rnorm()] for sampling from a normal distribution.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' nv_dnorm(x)
#' nv_dnorm(x, mean = 1, sd = 2)
#' nv_dnorm(x, log = TRUE)
#'
#' nv_pnorm(x)
#' nv_pnorm(x, mean = 1, sd = 2)
#' nv_pnorm(x, lower_tail = FALSE)
#' nv_pnorm(x, log_p = TRUE)
NULL

#' @rdname nv_normal
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

#' @rdname nv_normal
#' @export
#' @jit static c("lower_tail", "log_p")
nv_pnorm <- function(q, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE) {
  assert_flag(lower_tail)
  assert_flag(log_p)
  args <- as_anvl_arrays(q, mean, sd)
  q <- args[[1L]]
  mean <- args[[2L]]
  sd <- args[[3L]]
  op_dtype <- dtype(q)
  mean <- nv_convert(mean, op_dtype)
  sd <- nv_convert(sd, op_dtype)

  # Standardise, flipping sign if computing upper tail
  d <- if (lower_tail) (q - mean) / sd else (mean - q) / sd

  if (!log_p) {
    # When not computing log cdf we're done as no accuracy concerns with erfc
    return(0.5 * nv_erfc(-d / sqrt(2)))
  }

  # Here computing log cdf: care required to ensure accuracy deep in the tails,
  # since there is no log version of erfc in XLA.
  # R handles this with a near-minimax approximation due to Cody
  # <doi:10.1090/S0025-5718-1969-0247736-4>, but this algorithm does not perform
  # well with XLA due to complicated rational-polynomial expression.
  # Instead use the classic successive integration by parts asymptotic expansion
  # from Abramowitz & Stegun, eq 26.2.12 p.932 <isbn:0-486-61272-4> (originally
  # due to Laplace? Also used by JAX) if the argument is in a region where
  # direct evaluation of log(erfc) would be inaccurate.

  # Thresholds between direct computation of erfc and the asymptotic expansion,
  # Q, for f32 and f64. These differ from JAX for accuracy.
  is_f32 <- op_dtype == "f32"
  lower_threshold <- if (is_f32) -11.9 else -20
  upper_threshold <- 0

  # Computation regime:
  #   d <= lower_threshold ... then we compute log Q(-d) using asymptotic
  #                            expansion
  #   d > upper_threshold  ... then we compute log(1-erfc(d/sqrt(2))). Note the
  #                            approximation -erfc(d/sqrt(2)) has catastrophic
  #                            loss of accuracy
  #   d in between         ... accuracy of log(erfc(-d/sqrt(2))) is fine

  # Compute Q(-d) for the asymptotic region, first clamping the value to protect
  # gradient from poisoning later
  d_asymp <- nv_max(-d, 1)
  d2_asymp <- d_asymp * d_asymp
  w <- 1 / d2_asymp
  # Compute just what is required for precision (confirmed if statement compiles
  # away during tracing)
  series_minus_1 <- if (is_f32) {
    w * (-1 + w * 3)
  } else {
    w * (-1 + w * (3 + w * (-15 + w * (105 + w * (-945 + w * (10395 + w * (-135135)))))))
  }
  log_pdf_term <- -0.5 * d2_asymp - 0.5 * base::log(2 * pi)

  # Check which regime (asymptotic, direct, upper tail)
  use_non_asymp <- d > lower_threshold
  use_direct <- use_non_asymp & d <= upper_threshold
  # Compute correct erfc(-d/sqrt(2)) or erfc(d/sqrt(2)), selecting on arg to
  # avoid multiple erfc evaluations
  erfc_arg <- nv_ifelse(use_direct, -d, d)
  erfc_res <- 0.5 * nv_erfc(erfc_arg / sqrt(2))
  # Clamp result to a safe value on other branches so gradient not poisoned on
  # log/log1p calls
  erfc_res_direct <- nv_ifelse(use_direct, erfc_res, 1)
  erfc_res_upper <- nv_ifelse(use_non_asymp, erfc_res, 0)
  # Compute final answer down all branches, returning correct branch for each
  # element
  nv_ifelse(
    use_direct,
    nv_log(erfc_res_direct),
    nv_ifelse(
      use_non_asymp,
      nv_log1p(-erfc_res_upper),
      log_pdf_term -
        nv_log(d_asymp) +
        if (is_f32) series_minus_1 else nv_log1p(series_minus_1)
    )
  )
}
