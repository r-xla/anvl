## Probability distributions ---------------------------------------------------

#' @title The Normal Distribution
#' @name nv_normal
#' @description
#' Density (`nv_dnorm`), distribution function (`nv_pnorm`),
#' quantile function (`nv_qnorm`), and random
#' generation (`nv_rnorm`) for the Normal distribution with mean `mean` and
#' standard deviation `sd`.
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`).
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`.
#' @param mean ([`arrayish`])\cr
#'   Mean of the distribution (scalar or same shape as `x`/`q`/`p`).
#' @param sd ([`arrayish`])\cr
#'   Standard deviation of the distribution (scalar or same shape as
#'   `x`/`q`/`p`). Must be positive, otherwise results are invalid.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the densities/probabilities are given as logarithms. For
#'   `nv_qnorm` this describes the input `p`.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le x)}; otherwise,
#'   \eqn{P(X > x)}.
#' @details
#' The Normal distribution has probability density function:
#' \deqn{f(x) = \frac{1}{\sigma\sqrt{2\pi}}
#'   \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}
#' where \eqn{\mu} is the mean and \eqn{\sigma} is the standard deviation.
#' The `mean` and `sd` are converted to the data type of `x`/`q`/`p`.
#'
#' `nv_pnorm` uses the asymptotic expansion from
#' `r xlamisc::cite_bib("abramowitz1964handbook")`, equation 26.2.12, in the
#' left tail when `log_p = TRUE` to maintain accuracy.
#'
#' `nv_qnorm` uses the same minimax rational approximation as
#' `r xlamisc::cite_bib("moshier1989methods")` (this is `ndtri` in the Cephes
#' library as used by JAX) for `f64`, and uses a new lower degree Remez minimax
#' rational approximation on the same intervals for `f32`.
#' @references
#' `r xlamisc::format_bib("abramowitz1964handbook", "moshier1989methods")`
#' @seealso [nv_rnorm()] for sampling from a normal distribution.
#' @return
#' `nv_dnorm()` and `nv_pnorm()` return an [`arrayish`] with the same shape and
#' data type as `x`/`q`.
#'
#' `nv_rnorm()` returns a `list()` of two [`arrayish`] elements: the updated
#' RNG state and the sampled values.
#'
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
#'
#' p <- nv_array(c(0.025, 0.5, 0.975))
#' nv_qnorm(p)
#' nv_qnorm(p, mean = 1, sd = 2)
#' nv_qnorm(p, lower_tail = FALSE)
#' nv_qnorm(nv_array(c(-700, -2, -0.1), dtype = "f64"), log_p = TRUE)
NULL

#' @rdname nv_normal
#' @export
#' @jit static "log"
nv_dnorm <- function(x, mean = 0, sd = 1, log = FALSE) {
  assert_flag(log)
  args <- as_anvl_arrays(x = x, mean = mean, sd = sd, .promote = promote_like("x"))
  x <- args$x
  mean <- args$mean
  sd <- args$sd

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
  args <- as_anvl_arrays(q = q, mean = mean, sd = sd, .promote = promote_like("q"))
  q <- args$q
  mean <- args$mean
  sd <- args$sd
  op_dtype <- dtype(q)

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

# Horner's method for polynomials, coefficients in decreasing power order.
# x can be vector, say length n.
# coefs can be:
#   - a vector length d for a single polynomial; or
#   - a list of d vectors, each length n, for a different polynomial for each x.
#     Note layout is by power, so `coef[[1L]]` holds all n highest power coefs,
#     `coef[[2L]]` holds all n second highest power coefs etc.
#     Hence *only* suitable if all polynomials of the same degree.
horner <- function(x, coefs) {
  Reduce(function(acc, coef) acc * x + coef, coefs[-1L], init = coefs[[1L]])
}

# P/Q rational polynomial coefficients (P = numerator, Q = denominator), highest
# power first.
# - central region covers p \in (e^{-2}, 1-e{-2}] and is poly in w^2 where
#           w = p - 1/2;
# - tail region covers all other p. Separates into `tail` for z < 8 and
#        `far_tail` for z >= 8. Both are poly in 1/z where z = sqrt(-2 log t)
#        and t = min(p, 1 - p)
#
# NOTE: efficient use of Map in `select_far()` inside `nv_qnorm` assumes that
#       `p_tail` and `p_far_tail`, as well as `q_tail` and `q_far_tail` are the
#       same length, so any future Remez refit must ensure this or change
#       `select_far()` (applies to f32 and f64)
#
# First f64 precision: these are the coefficients from Cephes, as used also by
# JAX
qnorm_f64_coefs <- list(
  p_central = c(
    -5.99633501014107895267e1,
    9.80010754185999661536e1,
    -5.66762857469070293439e1,
    1.39312609387279679503e1,
    -1.23916583867381258016
  ),
  q_central = c(
    1.0,
    1.95448858338141759834,
    4.67627912898881538453,
    8.63602421390890590575e1,
    -2.25462687854119370527e2,
    2.00260212380060660359e2,
    -8.20372256168333339912e1,
    1.59056225126211695515e1,
    -1.18331621121330003142
  ),
  p_tail = c(
    4.05544892305962419923,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4
  ),
  q_tail = c(
    1.0,
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4
  ),
  p_far_tail = c(
    3.23774891776946035970,
    6.91522889068984211695,
    3.93881025292474443415,
    1.33303460815807542389,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9
  ),
  q_far_tail = c(
    1.0,
    6.02427039364742014255,
    3.67983563856160859403,
    1.37702099489081330271,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9
  )
)

# Then we specialise to f32: the above polynomials are overkill at f32 so below
# is an independent Remez fit for anvl using the same thresholds between
# central/tail/far tail and the same poly argument (w^2 or 1/z)
qnorm_f32_coefs <- list(
  p_central = c(-6.691131842723991e-1, 7.5626636219604695, -5.770283790138877, 1.047197585894062),
  q_central = c(-1.257612301180524e1, 1.820651998768941e1, -7.70932529281657, 1.0),
  p_tail = c(-1.1703880518959358, 9.77404924657488, 2.8949524675071373e1, 9.415665982832321, 9.171604050864e-1),
  q_tail = c(2.662973999005499e1, 1.0071629918518465e1, 1.0),
  p_far_tail = c(
    -1.3985698840384828e2,
    4.453781244880416e2,
    8.742497845723311e2,
    8.258251477108792e1,
    9.189365211474885e-1
  ),
  q_far_tail = c(9.349271395441176e2, 8.984706461134404e1, 1.0)
)

#' @rdname nv_normal
#' @export
#' @jit static c("lower_tail", "log_p")
nv_qnorm <- function(p, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE) {
  assert_flag(lower_tail)
  assert_flag(log_p)
  args <- as_anvl_arrays(p = p, mean = mean, sd = sd, .promote = promote_like("p"))
  p <- args$p
  mean <- args$mean
  sd <- args$sd
  op_dtype <- dtype(p)

  is_f32 <- op_dtype == "f32"

  cf <- if (is_f32) qnorm_f32_coefs else qnorm_f64_coefs
  lp <- if (log_p) p else nv_log(p)

  # As described above for rational polynomial coefficients, we divide into
  # regions.
  # upper tail if p > 1-e^-2         poly in 1/z where z = sqrt(-2 log p)
  # central    if e^-2 < p <= 1-e^-2 poly in w^2, w = p-0.5
  # lower tail if p <= e^-2          poly in 1/z where z = sqrt(-2 log (1-p))
  # Will actually handle lower tail by folding into upper via z = sqrt(-2 log t)
  # for t = min(p, 1-p).
  # The tail approximation is split between a near (z < 8) and far (z >= 8).

  # First, identify flags for upper and central region.
  # Then,
  #          use_upper  use_central
  # upper     TRUE       FALSE
  # central   FALSE      TRUE
  # lower     FALSE      FALSE
  if (log_p) {
    upper_threshold <- base::log1p(-exp(-2))
    use_upper <- lp > upper_threshold
    use_central <- (lp > -2) & (lp <= upper_threshold)
  } else {
    upper_threshold <- 1 - exp(-2)
    use_upper <- p > upper_threshold
    use_central <- (p > exp(-2)) & (p <= upper_threshold)
  }

  # Tail approximation
  # Compute log(1-p), with guards for derivatives ...
  log_t <- if (log_p) {
    nv_log(-nv_expm1(nv_ifelse(use_upper, lp, -1)))
  } else {
    nv_log1p(-nv_ifelse(use_upper, p, 0))
  }
  # ... and then log t = min(log p, log(1-p)) by selection
  log_t <- nv_ifelse(use_upper, log_t, lp)
  is_boundary <- log_t == -Inf
  # Safely clamp central region and boundary elements onto the branch boundary,
  # where tail is well behaved for gradients
  log_t <- nv_ifelse(is_boundary | use_central, -2, log_t)

  # Compute the near or far tail rational polynomial approximation
  # Poly is in 1/z for z = sqrt(-2 log t)
  z <- nv_sqrt(-2 * log_t)
  inv_z <- 1 / z
  use_far_tail <- z >= 8
  # See important "NOTE" preceding coefficients above regarding this helper func
  select_far <- function(far, near) {
    Map(function(x, y) nv_ifelse(use_far_tail, x, y), far, near)
  }
  ratio <- horner(inv_z, select_far(cf$p_far_tail, cf$p_tail)) /
    horner(inv_z, select_far(cf$q_far_tail, cf$q_tail))
  res_tail <- z - nv_log(z) * inv_z - ratio * inv_z

  # Central approximation
  # Poly is in w^2 for w = p-0.5 (accounting for if arg was log_p)
  w <- if (log_p) 0.5 * nv_expm1(lp + base::log(2)) else p - 0.5
  w2 <- w * w
  res_central <- base::sqrt(2 * pi) *
    (w + w * w2 * (horner(w2, cf$p_central) / horner(w2, cf$q_central)))

  # Final standardised Normal result
  # Distinguish central region from a tail, then resolve left/right tail
  res_std <- nv_ifelse(
    use_central,
    res_central,
    nv_ifelse(use_upper, res_tail, -res_tail)
  )
  res_std <- nv_ifelse(is_boundary, nv_ifelse(use_upper, Inf, -Inf), res_std)
  # Handle tail switch
  if (!lower_tail) {
    res_std <- -res_std
  }
  # Unstandardise as necessary
  mean + sd * res_std
}

#' @title The Uniform Distribution
#' @name nv_uniform
#' @description
#' Density (`nv_dunif`), distribution function (`nv_punif`), and quantile
#' function (`nv_qunif`) for the Uniform distribution on the interval from
#' `min` to `max`.
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`).
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`.
#' @param min,max ([`arrayish`])\cr
#'   Lower and upper limits of the distribution. Either scalars, or arrays of
#'   exactly the same shape as `x`/`q`/`p`, in which case the interval varies
#'   elementwise and each element of `x`/`q`/`p` is evaluated against its own
#'   `min`/`max`.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the densities/probabilities are given as logarithms. For
#'   `nv_qunif` this describes the input `p`.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le x)}; otherwise,
#'   \eqn{P(X > x)}.
#' @details
#' The Uniform distribution has probability density function:
#' \deqn{f(x) = \frac{1}{b - a}, \quad a \le x \le b}
#' and zero elsewhere, where \eqn{a} is `min` and \eqn{b} is `max`.
#' The `min` and `max` are converted to the data type of `x`/`q`/`p`.
#'
#' All three are univariate functions evaluated elementwise, returning one
#' value per element of `x`/`q`/`p`. Non-scalar `min`/`max` therefore give a
#' separate univariate Uniform per element, *not* a multivariate Uniform over
#' the hyper-rectangle \eqn{\prod_i [a_i, b_i]}. For that, reduce over the
#' result: `nv_reduce_prod(nv_dunif(x, min, max))`, or
#' `nv_reduce_sum(nv_dunif(x, min, max, log = TRUE))` on the log scale.
#'
#' @seealso [nv_runif()] for sampling from a uniform distribution.
#' @return
#' `nv_dunif()`, `nv_punif()`, and `nv_qunif()` return an [`arrayish`] with the
#' same shape and data type as `x`/`q`/`p`.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-0.5, 0, 0.25, 1, 1.5))
#' nv_dunif(x)
#' nv_dunif(x, min = -1, max = 2)
#' nv_dunif(x, log = TRUE)
#'
#' # `min`/`max` may vary elementwise, giving one univariate Uniform per
#' # element rather than a single distribution over a hyper-rectangle
#' lower <- nv_array(c(-1, -1, 0, 0, 1))
#' upper <- nv_array(c(0, 1, 1, 2, 2))
#' nv_dunif(x, min = lower, max = upper)
#'
#' nv_punif(x)
#' nv_punif(x, min = -1, max = 2)
#' nv_punif(x, lower_tail = FALSE)
#' nv_punif(x, log_p = TRUE)
#'
#' p <- nv_array(c(0.025, 0.5, 0.975))
#' nv_qunif(p)
#' nv_qunif(p, min = -1, max = 2)
#' nv_qunif(p, lower_tail = FALSE)
#' nv_qunif(nv_array(c(-700, -2, -0.1), dtype = "f64"), log_p = TRUE)
NULL

#' @rdname nv_uniform
#' @export
#' @jit static "log"
nv_dunif <- function(x, min = 0, max = 1, log = FALSE) {
  assert_flag(log)
  args <- as_anvl_arrays(x, min, max)
  x <- args[[1L]]
  min <- args[[2L]]
  max <- args[[3L]]
  op_dtype <- dtype(x)
  min <- nv_convert(min, op_dtype)
  max <- nv_convert(max, op_dtype)

  # Density constant on support, just need support indicator
  in_support <- (x >= min) & (x <= max)
  width <- max - min

  density <- if (log) {
    nv_ifelse(in_support, -nv_log(width), -Inf)
  } else {
    nv_ifelse(in_support, 1 / width, 0)
  }
  # NOTE: `in_support` will eval to FALSE when x is NaN, so need to restore a
  #       NaN result there. Similarly, the `max > min` check ensures NaN is
  #       restored for same reason if either is NaN while also rejecting
  #       reversed interval ends
  nv_ifelse(!nv_is_nan(x) & (max > min), density, NaN)
}

#' @rdname nv_uniform
#' @export
#' @jit static c("lower_tail", "log_p")
nv_punif <- function(q, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE) {
  assert_flag(lower_tail)
  assert_flag(log_p)
  args <- as_anvl_arrays(q, min, max)
  q <- args[[1L]]
  min <- args[[2L]]
  max <- args[[3L]]
  op_dtype <- dtype(q)
  min <- nv_convert(min, op_dtype)
  max <- nv_convert(max, op_dtype)

  width <- max - min
  # Resolve q against the endpoints before dividing, to match base R behaviour.
  # Also avoids degenerate 0 / 0 for edge case q == min == max.
  at_or_above <- q >= max
  at_or_below <- q <= min
  resolve_ends <- function(above_val, below_val, interior_val) {
    nv_ifelse(at_or_above, above_val, nv_ifelse(at_or_below, below_val, interior_val))
  }

  # Ensure all branches have safe value for gradients
  q_int <- nv_ifelse(at_or_above | at_or_below, min, q)

  u <- if (lower_tail) {
    resolve_ends(1, 0, (q_int - min) / width)
  } else {
    resolve_ends(0, 1, (max - q_int) / width)
  }

  # Reversed/non-finite interval is NaN to match base R: `valid` flag to track
  valid <- nv_is_finite(min) & nv_is_finite(max) & (max >= min)

  if (!log_p) {
    return(nv_ifelse(valid, u, NaN))
  }

  # To maintain accuracy of log near 1, switch to log1p in opposite tail mid way
  v <- if (lower_tail) {
    resolve_ends(0, 1, (max - q_int) / width)
  } else {
    resolve_ends(1, 0, (q_int - min) / width)
  }
  # So flag if can use log, else switch to log1p() of the opposite tail
  use_log <- u <= 0.5
  # Include inner clamp of a safe input on branch not taken for gradient calcs
  res <- nv_ifelse(
    use_log,
    nv_log(nv_ifelse(use_log, u, 1)),
    nv_log1p(-nv_ifelse(use_log, 0, v))
  )
  nv_ifelse(valid, res, NaN)
}

#' @rdname nv_uniform
#' @export
#' @jit static c("lower_tail", "log_p")
nv_qunif <- function(p, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE) {
  assert_flag(lower_tail)
  assert_flag(log_p)
  args <- as_anvl_arrays(p, min, max)
  p <- args[[1L]]
  min <- args[[2L]]
  max <- args[[3L]]
  op_dtype <- dtype(p)
  min <- nv_convert(min, op_dtype)
  max <- nv_convert(max, op_dtype)

  # Out-of-range `p` is resolved to NaN by `valid`, but also need `p_safe` to
  # avoid poisoning gradients
  if (log_p) {
    in_range <- p <= 0
    p_safe <- nv_ifelse(in_range, p, 0)
    u <- if (lower_tail) nv_exp(p_safe) else -nv_expm1(p_safe)
  } else {
    in_range <- (p >= 0) & (p <= 1)
    p_safe <- nv_ifelse(in_range, p, 0)
    u <- if (lower_tail) p_safe else 1 - p_safe
  }

  # Conditions to match NaN behaviour of base R
  valid <- in_range & nv_is_finite(min) & nv_is_finite(max) & (max >= min)
  nv_ifelse(valid, min + u * (max - min), NaN)
}
