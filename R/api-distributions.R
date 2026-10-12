#' @include jit.R
NULL

## Probability distributions ---------------------------------------------------

# The operand -- `x`/`q`/`p`, always the first argument -- governs the data type
# the distribution is evaluated at; the parameters follow it. The float check
# runs before the promotion, so a non-float operand is reported as itself rather
# than as a failure to bring a parameter to its data type.
promote_distribution_args <- function(...) {
  args <- list(...)
  operand <- names(args)[[1L]]
  assert_float_dtype(
    peek_dtype(args[[1L]]),
    arg = operand,
    hint = "Convert it with `nv_convert()`."
  )
  do.call(as_anvl_arrays, c(args, list(.promote = promotion_like(operand))))
}

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
#'   Mean of the distribution. Either a scalar, or an array of exactly the
#'   shape of `x`/`q`/`p` (or the sample, for `nv_rnorm`).
#' @param sd ([`arrayish`])\cr
#'   Standard deviation of the distribution, shaped like `mean`. Must be
#'   positive, otherwise results are invalid.
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
#'
#' `nv_pnorm` uses the asymptotic expansion from
#' `r cite_bib("abramowitz1964handbook")`, equation 26.2.12, in the
#' left tail when `log_p = TRUE` to maintain accuracy.
#'
#' `nv_qnorm` uses the same minimax rational approximation as
#' `r cite_bib("moshier1989methods")` (this is `ndtri` in the Cephes
#' library as used by JAX) for `f64`, and uses a new lower degree Remez minimax
#' rational approximation on the same intervals for `f32`.
#'
#' @templateVar dist norm
#' @templateVar params `mean` or `sd`
#' @template section_distribution_dtype
#' @references
#' `r format_bib("abramowitz1964handbook", "moshier1989methods")`
#' @seealso [nv_rnorm()] for sampling from a normal distribution.
#' @return ([`arrayish`] | named `list` of two [`arrayish`])\cr
#' `nv_dnorm()`, `nv_pnorm()` and `nv_qnorm()` return an [`arrayish`] with the
#' shape and data type of `x`/`q`/`p`.
#'
#' `nv_rnorm()` returns a named `list` of two [`arrayish`]: `state`, the updated
#' RNG state with the input `state`'s data type and shape, and `values`, the
#' sample of shape `shape` and the data type described under `dtype`.
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
nv_dnorm <- jit(
  function(x, mean = 0, sd = 1, log = FALSE) {
    assert_flag(log)
    args <- promote_distribution_args(x = x, mean = mean, sd = sd)
    x <- args$x
    mean <- args$mean
    sd <- args$sd

    z <- (x - mean) / sd
    log_density <- -0.5 * (z * z) - nv_log(sd) - 0.5 * base::log(2 * pi)

    if (log) {
      return(log_density)
    }
    nv_exp(log_density)
  },
  static = "log"
)

#' @rdname nv_normal
#' @export
nv_pnorm <- jit(
  function(q, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(q = q, mean = mean, sd = sd)
    q <- args$q
    mean <- args$mean
    sd <- args$sd
    # One threshold set per width, so a narrower float has none: it would
    # silently take the `f64` set.
    op_dtype <- assert_rng_float_dtype(dtype(q), arg = "q")

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
    d_asymp <- nv_pmax(-d, 1)
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
  },
  static = c("lower_tail", "log_p")
)

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
nv_qnorm <- jit(
  function(p, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(p = p, mean = mean, sd = sd)
    p <- args$p
    mean <- args$mean
    sd <- args$sd
    # One coefficient set per width -- see `nv_pnorm()`.
    op_dtype <- assert_rng_float_dtype(dtype(p), arg = "p")

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
      # The coefficients are plain R numbers, so a bare `nv_ifelse(pred, x, y)`
      # would have nothing to yield to and materialize at the default float,
      # dragging the whole result up with it. They are built at `p`'s data type
      # instead.
      Map(
        function(x, y) nv_ifelse(use_far_tail, nv_scalar_like(p, x), nv_scalar_like(p, y)),
        far,
        near
      )
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
    # The infinities are built at `p`'s data type: two bare R doubles here would
    # have nothing to yield to, materialize at the default float, and drag the
    # result up with them.
    res_std <- nv_ifelse(
      is_boundary,
      nv_ifelse(use_upper, nv_scalar_like(p, Inf), nv_scalar_like(p, -Inf)),
      res_std
    )
    # Handle tail switch
    if (!lower_tail) {
      res_std <- -res_std
    }
    # Unstandardise as necessary
    mean + sd * res_std
  },
  static = c("lower_tail", "log_p")
)

#' @title The Uniform Distribution
#' @name nv_uniform
#' @description
#' Density (`nv_dunif`), distribution function (`nv_punif`), quantile
#' function (`nv_qunif`), and random generation (`nv_runif`) for the Uniform
#' distribution on the interval from `min` to `max`.
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`).
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`.
#' @param min,max ([`arrayish`])\cr
#'   Lower and upper limits of the distribution. Either scalars, or arrays of
#'   exactly the same shape as `x`/`q`/`p` (or the sample, for `nv_runif`), in
#'   which case the interval varies elementwise and each element of `x`/`q`/`p`
#'   is evaluated against, or each draw made from, its own `min`/`max`.
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
#'
#' All four are univariate functions evaluated elementwise, returning one
#' value per element of `x`/`q`/`p` (or of the sample). Non-scalar `min`/`max` therefore give a
#' separate univariate Uniform per element, *not* a multivariate Uniform over
#' the hyper-rectangle \eqn{\prod_i [a_i, b_i]}. For that, reduce over the
#' result: `nv_prod(nv_dunif(x, min, max))`, or
#' `nv_sum(nv_dunif(x, min, max, log = TRUE))` on the log scale.
#'
#' @templateVar dist unif
#' @templateVar params `min` or `max`
#' @template section_distribution_dtype
#' @return ([`arrayish`] | named `list` of two [`arrayish`])\cr
#' `nv_dunif()`, `nv_punif()`, and `nv_qunif()` return an [`arrayish`] with the
#' same shape and data type as `x`/`q`/`p`.
#'
#' `nv_runif()` returns a named `list` of two [`arrayish`]: `state`, the updated
#' RNG state with the input `state`'s data type and shape, and `values`, the
#' sample of shape `shape` and the data type described under `dtype`.
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
#'
#' # `state` is the updated RNG state, `values` the sample
#' state <- nv_rng_state(42L)
#' result <- nv_runif(c(2, 3), state)
#' result$values
#'
#' # `min`/`max` may also be arrays of the same shape as the sample
#' lower <- nv_array(matrix(c(0, 10, 20, 30, 40, 50), nrow = 2))
#' nv_runif(c(2, 3), state, min = lower, max = lower + 1)$values
NULL

#' @rdname nv_uniform
#' @export
nv_dunif <- jit(
  function(x, min = 0, max = 1, log = FALSE) {
    assert_flag(log)
    args <- promote_distribution_args(x = x, min = min, max = max)
    x <- args$x
    min <- args$min
    max <- args$max

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
  },
  static = "log"
)

#' @rdname nv_uniform
#' @export
nv_punif <- jit(
  function(q, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(q = q, min = min, max = max)
    q <- args$q
    min <- args$min
    max <- args$max

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
  },
  static = c("lower_tail", "log_p")
)

#' @rdname nv_uniform
#' @export
nv_qunif <- jit(
  function(p, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(p = p, min = min, max = max)
    p <- args$p
    min <- args$min
    max <- args$max

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
  },
  static = c("lower_tail", "log_p")
)

# log(1 - exp(x)) for x <= 0, switching at -log(2) between the two forms that
# keep full accuracy either side of it (Maechler, "Accurately Computing
# log(1 - exp(-|a|))"; R's `R_Log1_Exp()`). Each form is fed a safe stand-in on
# the elements where it is not selected, so it cannot poison the gradient.
log1mexp <- function(x) {
  use_expm1 <- x > -base::log(2)
  nv_ifelse(
    use_expm1,
    nv_log(-nv_expm1(nv_ifelse(use_expm1, x, -1))),
    nv_log1p(-nv_exp(nv_ifelse(use_expm1, -1, x)))
  )
}

# Whether `rate` is admissible. base R tests the scale 1 / rate, so -0 (scale
# -Inf) is rejected along with negative and NaN rates. Only zero needs the
# reciprocal to tell its sign: taking it of every rate would reject large rates
# whose reciprocal flushes to zero. `nv_floor()` keeps the sign of zero and has a
# zero derivative, which stops the reverse pass forming 0 * Inf through 1 / 0.
valid_exp_rate <- function(rate) {
  zero <- rate == 0
  (rate >= 0) & !(zero & (1 / nv_floor(nv_ifelse(zero, rate, 1)) < 0))
}

#' @title The Exponential Distribution
#' @name nv_exponential
#' @description
#' Density (`nv_dexp`), distribution function (`nv_pexp`), and quantile
#' function (`nv_qexp`) for the Exponential distribution with rate `rate`
#' (i.e., mean `1 / rate`).
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`).
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`. With `log_p = TRUE`, supply log-probabilities
#'   in \eqn{[-\infty, 0]} instead.
#' @param rate ([`arrayish`])\cr
#'   Rate of the distribution. Scalars broadcast; otherwise `rate` and
#'   `x`/`q`/`p` must have the same shape. Negative rates (including `-Inf`
#'   and negative zero) give `NaN`.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the densities/probabilities are given as logarithms. For
#'   `nv_qexp` this describes the input `p`.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le x)}; otherwise,
#'   \eqn{P(X > x)}.
#' @details
#' The Exponential distribution has probability density function:
#' \deqn{f(x) = \lambda e^{-\lambda x}, \quad x \ge 0}
#' and zero elsewhere, where \eqn{\lambda} is `rate`.
#'
#' For positive finite rates, the mean is `1 / rate`. Zero and infinite rates
#' follow base R's endpoint conventions:
#' * `rate = 0`: the density and lower-tail probability are zero at finite
#'   quantiles, but `NaN` at infinity. Quantiles are infinite except at
#'   lower-tail probability zero, where the result is zero.
#' * `rate = Inf`: densities are `NaN`. The lower-tail probability is zero
#'   at or below zero and one above zero. Quantiles are zero except at
#'   lower-tail probability one, where the result is `NaN`.
#'
#' @templateVar dist exp
#' @templateVar params `rate`
#' @template section_distribution_dtype
#' @return ([`arrayish`])\cr
#' `nv_dexp()`, `nv_pexp()`, and `nv_qexp()` return an [`arrayish`] with the
#' data type of `x`/`q`/`p` and the shape after scalar broadcasting with `rate`.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 0.5, 1, 2))
#' nv_dexp(x)
#' nv_dexp(x, rate = 2)
#' nv_dexp(x, log = TRUE)
#'
#' nv_pexp(x)
#' nv_pexp(x, rate = 2)
#' nv_pexp(x, lower_tail = FALSE)
#' nv_pexp(x, log_p = TRUE)
#'
#' p <- nv_array(c(0.025, 0.5, 0.975))
#' nv_qexp(p)
#' nv_qexp(p, rate = 2)
#' nv_qexp(p, lower_tail = FALSE)
#' nv_qexp(nv_array(c(-700, -2, -0.1), dtype = "f64"), log_p = TRUE)
NULL

#' @rdname nv_exponential
#' @export
nv_dexp <- jit(
  function(x, rate = 1, log = FALSE) {
    assert_flag(log)
    args <- promote_distribution_args(x = x, rate = rate)
    x <- args$x
    rate <- args$rate
    op_dtype <- assert_rng_float_dtype(dtype(x), arg = "x")

    # Below the support and at `x = Inf` the density is resolved directly, and
    # `x` is clamped so the untaken branch cannot poison gradients.
    # NaN fails both comparisons and flows through.
    resolved <- x < 0 | x == Inf
    x_safe <- nv_ifelse(resolved, 0, x)
    density <- if (log) {
      nv_ifelse(resolved, -Inf, nv_log(rate) - rate * x_safe)
    } else {
      # Splitting exponent delays underflow so rate can rescale result first.
      t <- -rate * x_safe
      split <- t < (if (op_dtype == "f32") -80 else -700)
      e <- nv_exp(nv_ifelse(split, t / 2, t))
      scaled <- rate * e
      nv_ifelse(resolved, 0, nv_ifelse(split, scaled * e, scaled))
    }
    valid <- valid_exp_rate(rate) & (rate != Inf) & !((x == Inf) & (rate == 0))
    nv_ifelse(valid, density, NaN)
  },
  static = "log"
)

#' @rdname nv_exponential
#' @export
nv_pexp <- jit(
  function(q, rate = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(q = q, rate = rate)
    q <- args$q
    rate <- args$rate
    op_dtype <- assert_rng_float_dtype(dtype(q), arg = "q")

    # At or below the support and at `q = Inf` the cdf is resolved directly,
    # which also keeps `rate = Inf` well defined at `q = 0`. `q` is clamped
    # there so untaken branch cannot poison gradients.
    # NaN fails both comparisons and flows through.
    # NB: need to treat subnormal q as zero explicitly so compiler optimizations
    #     can't cause the value and gradient guards to disagree.
    #     Hence at_or_below not simply `q <= 0`
    at_or_below <- q < (if (op_dtype == "f32") 2^-126 else 2^-1022)
    at_inf <- q == Inf
    resolve_ends <- function(below_val, inf_val, interior_val) {
      nv_ifelse(at_or_below, below_val, nv_ifelse(at_inf, inf_val, interior_val))
    }
    # compute exponent, with a guard value outside support or at inf
    t <- -rate * nv_ifelse(at_or_below | at_inf, 1, q)

    res <- if (lower_tail) {
      # => 1-e^t
      if (log_p) {
        # => log(1-e^t)
        # If rate & q are not subnormal, but rate * q underflows, then t flushes
        # to zero and log1mexp() gives -Inf.
        # But log(1 - exp(-rate * q)) is approx log(rate * q) for tiny rate*q so
        # this should be computable.
        # Hence scale each rate & q by 2^k (half the exponent range) to avoid
        # subnormal product without overflowing, then rescale back.
        # Only split near underflow: log1mexp() more accurate otherwise.
        underflow <- (t == 0) & (rate > 0) & !at_or_below & !at_inf
        k <- if (op_dtype == "f32") 63 else 511
        # Select after scaling each factor so XLA cannot reassociate this as
        # (rate * q) * 2^(2*k), underflowing before the scale is applied.
        rescaled <- nv_log(
          nv_ifelse(underflow, rate * 2^k, 1) * nv_ifelse(underflow, q * 2^k, 1)
        ) -
          2 * k * base::log(2)
        interior <- nv_ifelse(underflow, rescaled, log1mexp(nv_ifelse(underflow, -1, t)))
        resolve_ends(-Inf, 0, interior)
      } else {
        resolve_ends(0, 1, -nv_expm1(t))
      }
    } else {
      # => e^t
      if (log_p) resolve_ends(0, -Inf, t) else resolve_ends(1, 0, nv_exp(t))
    }
    # Resolve to NaN matching base R rules
    valid <- valid_exp_rate(rate) & !(at_inf & (rate == 0))
    nv_ifelse(valid, res, NaN)
  },
  static = c("lower_tail", "log_p")
)

#' @rdname nv_exponential
#' @export
nv_qexp <- jit(
  function(p, rate = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(p = p, rate = rate)
    p <- args$p
    rate <- args$rate
    op_dtype <- assert_rng_float_dtype(dtype(p), arg = "p")

    # Valid range checks for p, gradient guarding and exact zero result resolution
    if (log_p) {
      in_range <- p <= 0
      p_safe <- nv_ifelse(in_range, p, -1)
      zero_result <- p == (if (lower_tail) -Inf else 0)
    } else {
      in_range <- (p >= 0) & (p <= 1)
      p_safe <- nv_ifelse(in_range, p, 0.5)
      zero_result <- p == (if (lower_tail) 0 else 1)
    }

    # If degenerate case (zero-quantile endpoint and rate 0 => 0/0) resolve
    # directly, with gradient guard.
    degenerate <- zero_result & (rate == 0)
    rate_safe <- nv_ifelse(degenerate | !in_range, 1, rate)

    quantile <- if (lower_tail && log_p) {
      # => -log(1-e^p)/rate
      # For small exp(p), -log(1 - exp(p)) is approx exp(p). But if exp(p)
      # underflows (or subnormal) result lost before division by rate.
      # Hence, split exponential itself so rate can rescale before underflow,
      # but avoiding costly log(rate) calculation.
      # (exp(p/2)/rate)*exp(p/2) = exp(p)/rate
      # Only split near underflow: log1mexp() more accurate otherwise.
      underflow <- p_safe < (if (op_dtype == "f32") -80 else -700)
      half <- nv_exp(nv_ifelse(underflow, p_safe / 2, -1))
      zero_rate <- rate_safe == 0
      rescaled <- nv_ifelse(zero_rate, Inf, (half / nv_ifelse(zero_rate, 1, rate_safe)) * half)
      regular <- -log1mexp(nv_ifelse(underflow, -1, p_safe)) / rate_safe
      nv_ifelse(underflow, rescaled, regular)
    } else {
      # Remaining easy cases
      log_upper <- if (lower_tail) {
        nv_log1p(-p_safe)
      } else {
        if (log_p) p_safe else nv_log(p_safe)
      }
      -log_upper / rate_safe
    }
    # Resolve to NaN matching base R rules
    valid <- in_range & valid_exp_rate(rate)
    nv_ifelse(valid, nv_ifelse(degenerate, 0, quantile), NaN)
  },
  static = c("lower_tail", "log_p")
)

# Whether `x` is not a whole number, with base R's tolerance (`R_nonint()`).
# Infinities count as whole, as in base R.
is_nonint <- function(x) {
  nv_abs(x - nv_round(x)) > 1e-9 * nv_pmax(nv_abs(x), 1)
}

# Computes c - a * b with less rounding error when c is close to a * b.
# Splits a and b into high and low parts using bit operations, then subtracts
# their partial products separately. The result is not guaranteed exact.
#
# Adapted from Dekker's splitting technique:
# T. J. Dekker (1971), "A floating-point technique for extending the available
# precision", Numerische Mathematik 18, 224–242.
# https://doi.org/10.1007/BF01397083
#
# Use bit operations for the split because XLA could simplify away an
# arithmetic splitting method based on multiplication by a constant.
# Bitcasts carry no gradient, but the remainders a - a_hi and b - b_hi
# preserve the derivatives needed for c - a * b.
sub_exact_prod <- function(c, a, b) {
  is_f32 <- dtype(a) == "f32"
  bits_dtype <- if (is_f32) "ui32" else "ui64"
  drop <- if (is_f32) 12L else 27L
  head <- function(x) {
    bits <- nv_bitcast_convert(x, bits_dtype)
    nv_bitcast_convert(nv_shift_left(nv_shift_right_logical(bits, drop), drop), dtype(x))
  }
  a_hi <- head(a)
  a_lo <- a - a_hi
  b_hi <- head(b)
  b_lo <- b - b_hi
  (((c - a_hi * b_hi) - a_hi * b_lo) - a_lo * b_hi) - a_lo * b_lo
}

# Stirling's formula error, log(n!) - log(sqrt(2 pi n) (n / e)^n), for whole
# n >= 1, as in R's `stirlerr()`. Up to n = 15 the exact values are tabulated;
# above that the series in 1 / n^2 has converged to double precision by its
# seventh term, which is the term count R uses.
binom_stirlerr <- function(n) {
  series <- horner(1 / (n * n), stirlerr_series) / n
  Reduce(
    function(acc, i) nv_ifelse(n == i, stirlerr_table[[i]], acc),
    seq_along(stirlerr_table),
    init = series
  )
}

# 1/12 - 1/(360 n^2) + 1/(1260 n^4) - ..., as a polynomial in 1 / n^2, highest
# power first
# See `src/nmath/stirlerr.c` in R source code for values in `stirlerr_series`
# and `stirlerr_table`
stirlerr_series <- c(1 / 156, -691 / 360360, 1 / 1188, -1 / 1680, 1 / 1260, -1 / 360, 1 / 12)

# Only whole-number entries from R's sferr_halves table are needed here:
# the binomial calls stirlerr at n, x, and n - x, all positive whole numbers.
# Revisit before reusing this helper for other distributions: Student's t
# needs half-integer (and, for non-whole degrees of freedom, general real)
# arguments, which this helper is not designed to handle.
stirlerr_table <- c(
  0.0810614667953272582196702,
  0.0413406959554092940938221,
  0.02767792568499833914878929,
  0.02079067210376509311152277,
  0.01664469118982119216319487,
  0.01387612882307074799874573,
  0.01189670994589177009505572,
  0.010411265261972096497478567,
  0.009255462182712732917728637,
  0.008330563433362871256469318,
  0.007573675487951840794972024,
  0.006942840107209529865664152,
  0.006408994188004207068439631,
  0.005951370112758847735624416,
  0.005554733551962801371038690
)

# Compute x * log(x / np) + np - x for positive x and np, as in base R's
# internal bd0() function. This is a term in Loader's saddle-point formula
# for the binomial probability.
# The caller supplies d = x - n*p, calculated more accurately than subtracting
# the rounded product np from x.
#
# Near x = np, the direct formula subtracts nearly equal terms and loses
# precision. Instead, use a series in v = d / (x + np). We use it only when
# |v| < 0.1, where 4 correction terms suffice for f32 and 8 for f64.
binom_bd0 <- function(x, np, d, n_terms) {
  use_series <- nv_abs(d) < 0.1 * (x + np)

  v <- d / (x + np)
  w <- v * v
  # sum_{j = 1}^{n_terms} w^(j - 1) / (2 j + 1), highest power first
  tail <- horner(w, 1 / (2 * rev(seq_len(n_terms)) + 1))
  series <- d * v + 2 * x * v * w * tail

  # Use whichever of np / x and x / np is <= 1 as the log argument.
  # Both forms give the same result, but this choice helps preserve gradients:
  # log(r) has derivative 1/r, so a large r introduces a tiny factor that can
  # underflow when multiplied by an already tiny probability gradient.
  #
  # The binomial caller has 1 <= x <= n and np = n*p. For a normal positive p,
  # these bounds keep both ratios from overflowing or becoming subnormal.
  # The same holds for the failure counts, using 1-p instead of p.
  # XLA treats a subnormal p as zero, which the caller handles separately.
  direct <- nv_ifelse(x > np, np - x * (nv_log(np / x) + 1), x * nv_log(x / np) + np - x)

  nv_ifelse(use_series, series, direct)
}

# Log of the Binomial probability mass at whole `x`, for whole `n >= 0` and
# `0 <= p <= 1`, as in R's `dbinom_raw()` (Loader's saddle point algorithm).
# Avoids constructing huge factorials or subtracting large, nearly equal
# log-factorial expressions.
# Each branch is fed safe stand-ins on the elements where it is not selected, so
# that it cannot poison the gradient. The stand-ins for `n` and `p` are chosen
# by conditions on `n` and `p` alone, so a scalar `n` or `p` stays scalar.
binom_log_density <- function(x, n, p, n_terms) {
  # flags to identify cases later
  outside <- x < 0 | x > n | x == Inf
  zero_trials <- n == 0
  impossible <- (p == 0 & x > 0) | (p == 1 & x < n)
  at_zero <- x == 0
  at_size <- x == n
  infinite_trials <- n == Inf

  # Two edge cases, then interior case

  # x == 0: the log probability is n*log(1 - p).
  n_zero <- nv_ifelse(infinite_trials, 1, n)
  p_zero <- nv_ifelse(p < 1, p, 0.5)
  # XLA's CPU f64 log1p is off by up to ~120 ulp for arguments in
  # [1 - sqrt(2), -0.36], an error that n multiplies here (~1e-11 relative for
  # size = 1000, prob near 0.414). Fixed upstream in openxla/xla#46765, so
  # deliberately not worked around.
  log_zero <- nv_ifelse(infinite_trials & (p > 0), -Inf, n_zero * nv_log1p(-p_zero))

  # x == n: the log probability n*log(p).
  log_size <- n * nv_log(nv_ifelse(p > 0, p, 0.5))

  # 0 < x < n and 0 < p < 1, interior cases, so necessarily n >= 2.
  interior <- !(outside | zero_trials | impossible | at_zero | at_size | infinite_trials)
  # When not interior ensure safe inputs
  # The safety value x = 1 is inside the support of any safety choice for n.
  x_int <- nv_ifelse(interior, x, 1)
  n_int <- nv_ifelse((n >= 2) & !infinite_trials, n, 2)
  p_int <- nv_ifelse((p > 0) & (p < 1), p, 0.5)
  np <- n_int * p_int
  nq <- n_int * (1 - p_int)
  # (n - x) - nq is exactly -(x - np)
  d <- sub_exact_prod(x_int, n_int, p_int)
  # First three terms: Stirling factorial corrections for n!, x!, and (n-x)!
  # Last two terms: success and failure count deviance 
  lc <- binom_stirlerr(n_int) -
    binom_stirlerr(x_int) -
    binom_stirlerr(n_int - x_int) -
    binom_bd0(x_int, np, d, n_terms) -
    binom_bd0(n_int - x_int, nq, -d, n_terms)
  # log1p(-x / n) loses the digits of n - x when x is close to n, where that
  # difference is exact instead. For x / n in [0.36, sqrt(2) - 1] the
  # log1p argument falls in XLA's CPU f64 log1p defect band (up to ~120 ulp),
  # but only half of it enters the log-density, adding ~3e-15 absolute. Fixed
  # upstream in openxla/xla#46765, so deliberately not worked around.
  # log((n-x)/n) computed stabley
  log_frac <- nv_ifelse(
    x_int > 0.5 * n_int,
    nv_log(n_int - x_int) - nv_log(n_int),
    nv_log1p(-x_int / n_int)
  )
  # Log of the square-root prefactor from Stirling's formula
  lf <- base::log(2 * pi) + nv_log(x_int) + log_frac
  log_interior <- lc - 0.5 * lf

  nv_ifelse(
    outside | impossible,
    -Inf,
    nv_ifelse(
      zero_trials,
      0,
      nv_ifelse(
        at_zero,
        log_zero,
        nv_ifelse(at_size, log_size, nv_ifelse(infinite_trials, -Inf, log_interior))
      )
    )
  )
}

# x - log(1 + x) for |x| <= 0.03, as R's `rlog1()`. With r = x / (2 + x) it is
# r x - 2 (r^3 / 3 + r^5 / 5 + ...), free of the cancellation in the direct
# form; r^2 < 2.5e-4, so `n_terms` = 6 terms reach double precision, 3 single.
rlog1 <- function(x, n_terms) {
  r <- x / (2 + x)
  r2 <- r * r
  r * x - 2 * r * r2 * horner(r2, 1 / (2 * rev(seq_len(n_terms)) + 1))
}

# exp(z^2) erfc(z) for z >= 0: directly while erfc(z) is a normal number, and
# from its asymptotic series beyond `z_switch`, which has converged by `n_terms`
# terms there.
erfcx <- function(z, z_switch, n_terms) {
  use_series <- z > z_switch
  z_direct <- nv_ifelse(use_series, 0, z)
  z_series <- nv_ifelse(use_series, z, z_switch)
  u <- 1 / (2 * z_series * z_series)
  # 1 - u + 1 * 3 u^2 - 1 * 3 * 5 u^3 + ...
  series <- Reduce(
    function(acc, k) 1 - (2 * k - 1) * u * acc,
    rev(seq_len(n_terms)),
    init = 1
  )
  nv_ifelse(
    use_series,
    series / (z_series * base::sqrt(pi)),
    nv_exp(z_direct * z_direct) * nv_erfc(z_direct)
  )
}

# del(a) + del(b) - del(a + b) for a, b >= 8, where
# del(x) = log Gamma(x) - (x - 1/2) log(x) + x - log(2 pi) / 2, as R's `bcorr()`,
# which avoids the cancellation between del(b) and del(a + b).
bcorr <- function(a0, b0) {
  a <- nv_pmin(a0, b0)
  b <- nv_pmax(a0, b0)
  h <- a / b
  c <- h / (h + 1)
  x <- 1 / (h + 1)
  x2 <- x * x
  # s3, s5, ..., s11
  s <- Reduce(function(acc, i) c(acc, list(x + x2 * acc[[length(acc)]] + 1)), 1:4, init = list(x + x2 + 1))
  coefs <- c(
    0.0833333333333333,
    -0.00277777777760991,
    7.9365066682539e-4,
    -5.9520293135187e-4,
    8.37308034031215e-4,
    -0.00165322962780713
  )
  t <- 1 / (b * b)
  # The coefficients after the first are weighted by s3, ..., s11
  w <- horner(t, rev(c(list(coefs[[1L]]), Map(`*`, coefs[-1L], s))))
  w <- w * c / b
  t <- 1 / (a * a)
  horner(t, rev(coefs)) / a + w
}

# log I_x(a, b) from the asymptotic expansion of DiDonato & Morris (TOMS 708
# `basym()`, as in R's `pbeta()`) for large a and b, with
# lambda = a - (a + b) x >= 0 small relative to them. The coefficients are fixed
# recurrences, unrolled here; `n_terms` (even) is where the expansion is cut.
basym <- function(a, b, lambda, op_dtype) {
  is_f32 <- op_dtype == "f32"
  n_terms <- if (is_f32) 4L else 8L
  e0 <- 2 / base::sqrt(pi)
  e1 <- 2^-1.5

  rlog1_terms <- if (is_f32) 3L else 6L
  f <- nv_pmax(a * rlog1(-lambda / a, rlog1_terms) + b * rlog1(lambda / b, rlog1_terms), 0)
  z0 <- nv_sqrt(f)
  z <- z0 / e1 * 0.5
  z2 <- f + f
  a_lt_b <- a < b
  h <- nv_ifelse(a_lt_b, a / b, b / a)
  r0 <- 1 / (h + 1)
  r1 <- (b - a) / nv_pmax(a, b)
  w0 <- 1 / nv_sqrt(nv_pmin(a, b) * (h + 1))

  a0 <- list(r1 * 2 / 3)
  cc <- list(-a0[[1L]] / 2)
  dd <- list(a0[[1L]] / 2)
  j0 <- 0.5 / e0 * erfcx(z0, if (is_f32) 8 else 20, if (is_f32) 6L else 8L)
  j1 <- e1
  sum <- j0 + dd[[1L]] * w0 * j1

  s <- 1
  h2 <- h * h
  hn <- 1
  w <- w0
  znm1 <- z
  zn <- z2
  for (n in seq(2L, n_terms, by = 2L)) {
    hn <- hn * h2
    a0[[n]] <- r0 * 2 * (h * hn + 1) / (n + 2)
    s <- s + hn
    a0[[n + 1L]] <- r1 * 2 * s / (n + 3)
    for (i in c(n, n + 1L)) {
      r <- (i + 1) * -0.5
      b0 <- list(r * a0[[1L]])
      for (m in seq_len(i)[-1L]) {
        bsum <- Reduce(`+`, lapply(seq_len(m - 1L), function(j) (j * r - (m - j)) * a0[[j]] * b0[[m - j]]))
        b0[[m]] <- r * a0[[m]] + bsum / m
      }
      cc[[i]] <- b0[[i]] / (i + 1)
      dsum <- if (i > 1L) Reduce(`+`, lapply(seq_len(i - 1L), function(j) dd[[i - j]] * cc[[j]])) else 0
      dd[[i]] <- -(dsum + cc[[i]])
    }
    j0 <- e1 * znm1 + (n - 1) * j0
    j1 <- e1 * zn + n * j1
    znm1 <- z2 * znm1
    zn <- z2 * zn
    w <- w * w0
    t0 <- dd[[n]] * w * j0
    w <- w * w0
    t1 <- dd[[n + 1L]] * w * j1
    sum <- sum + t0 + t1
  }

  base::log(e0) - f - bcorr(a, b) + nv_log(sum)
}

# The Binomial distribution function at whole `k`, for whole `n` with
# 0 <= k < n and 0 < p < 1, through the incomplete beta function
# P(X > k) = I_p(k + 1, n - k), evaluated as in R's `pbeta()` (TOMS 708,
# DiDonato and Morris (1992)).
#
# For large beta parameters near the mean, we use the asymptotic expansion
# `basym()`. Otherwise we use the continued fraction from TOMS 708's `bfrac()`,
# which typically converges within a few dozen iterations.
# An element whose continued fraction has not converged within 1000 iterations
# is NaN.
binom_cdf <- function(k, n, p, lower_tail, log_p, op_dtype) {
  is_f32 <- op_dtype == "f32"
  eps <- if (is_f32) 2^-23 else 2^-52
  n_terms <- if (is_f32) 4L else 8L

  q <- 1 - p
  # lambda = (k + 1) - (n + 1) p with reduced error
  lambda <- sub_exact_prod(k + 1, n + 1, p)
  # Choose the beta-function orientation with lambda >= 0.
  # If upper is true, compute I_p(k + 1, n - k) = P(X > k).
  # Otherwise compute I_{1-p}(n - k, k + 1) = P(X <= k).
  # Complement taken later if the caller requests the other tail.
  upper <- lambda >= 0
  a <- nv_ifelse(upper, k + 1, n - k)
  b <- nv_ifelse(upper, n - k, k + 1)
  x <- nv_ifelse(upper, p, q)
  y <- nv_ifelse(upper, q, p)
  lambda <- nv_abs(lambda)

  ab_min <- nv_pmin(a, b)
  # Identify when to use asymptotic expansion method `basym()`
  use_asym <- (ab_min > 1000) & (lambda <= 0.03 * ab_min)
  # The expansion is long, and only needed for large a and b, so it is skipped
  # at run time where no element needs it.
  # A while loop that runs at most once lets us skip that work: XLA may lower
  # `nv_if()` into a select, which computes both branches.
  log_asym <- nv_while(
    init = list(pending = nv_any(use_asym), value = nv_fill_like(lambda, 0)),
    cond = function(pending, value) pending,
    body = function(pending, value) {
      list(
        pending = nv_scalar(FALSE),
        value = basym(
          # This runs for the whole array if any element needs the expansion.
          # Give the other elements safe inputs within the expansion's domain.
          nv_ifelse(use_asym, a, 2000),
          nv_ifelse(use_asym, b, 2000),
          nv_ifelse(use_asym, lambda, 0),
          op_dtype
        )
      )
    }
  )$value

  # Set up constants for the continued fraction.
  c <- lambda + 1
  c1 <- 1 / a + 1
  yp1 <- y + 1
  float_dtype <- dtype(lambda)

  # The continued fraction uses the previous and current numerator and
  # denominator. Between loop iterations, we divide all four by the current
  # denominator. This makes it 1 and the current numerator equal to the
  # approximation r, so we only need to keep three values: an, bn, and r.
  #
  # The other counters used by TOMS 708 can be calculated from the step
  # number m, which is shared by all array elements.
  #
  # Once an element converges, we keep its answer r. Its an and bn values
  # continue to change while other elements finish, but no longer affect
  # its answer.
  ab <- a + b
  # 1 / tau, see `step()`
  u <- 1 / (a + 1)

  # For positive normal x, read its exponent bits to find the power of two
  # that scales it into [1, 2). This avoids division. Scaling is exact while
  # the results remain normal; smaller values can still underflow. We use
  # this after each step; see the comments above `step()` for the limits.
  is_f32 <- op_dtype == "f32"
  bits_dtype <- if (is_f32) "ui32" else "ui64"
  n_mantissa <- if (is_f32) 23L else 52L
  exponent_max <- if (is_f32) 254L else 2046L
  inv_pow2 <- function(x) {
    e <- nv_shift_right_logical(nv_bitcast_convert(x, bits_dtype), n_mantissa)
    nv_bitcast_convert(nv_shift_left(exponent_max - e, n_mantissa), float_dtype)
  }
  # Advance the continued fraction by one step, using the TOMS 708 recurrence
  # (DiDonato and Morris, 1992).
  #
  # We rearrange its coefficients to avoid division inside the step. With
  # tau = a + 1 and s = a + 2 m - 1, we multiply the original beta coefficient
  # by k_m = s (s + 2) / tau^2, and alpha by k_m k_(m-1). Together these
  # changes preserve the value of the continued fraction. The first step
  # uses k_0 = 1 / tau, so its s - 2 factor in alpha is replaced by 1.
  #
  # Divisions by tau use the precomputed reciprocal u, spread across the
  # factors to keep intermediate values small. In alpha, we first multiply
  # (w * x) by ((a + b + m - 1) * x / tau), then apply the remaining three
  # factors. The chosen orientation gives (a + b) * x <= a, so that second
  # factor is at most 1 + m. This limits growth early in the product.
  #
  # If that initial product underflows, each remaining factor is at most
  # 1 + 2 m, so the lost alpha has magnitude at most roughly
  # tiny * (1 + 2 m)^3, where tiny is the smallest positive normal value.
  # This bounds the lost coefficient, not its effect on every later step;
  # in particular, w = m (b - m) becomes negative beyond m = b, so we cannot
  # assume positive recurrence coefficients throughout the unrolled steps.
  #
  # After each step, we scale both numerator/denominator pairs by the same
  # power of two. For a positive normal current denominator, this brings
  # it into [1, 2). Both ratios are preserved unless scaling loses values
  # to underflow. This controls growth without division.
  #
  # At extreme parameter values, the recurrence can still overflow before
  # rescaling, or the convergence test can underflow. These precautions do
  # not guarantee that every intermediate value remains within range.
  step <- function(m, an, anp1, bn, bnp1) {
    s <- a + 2 * m - 1
    w <- m * (b - m)
    s_minus_2 <- nv_ifelse(m == 1, 1, s - 2)
    # Apply the reciprocal u within each factor, in the order described above.
    su <- s * u
    s2u <- (s + 2) * u
    alpha <- (w * x) * ((ab + m - 1) * x * u) * ((a + m - 1) * u) * s2u * (s_minus_2 * u)
    beta <- w * x * s2u * u + m * su * s2u + ((a + m) * u) * su * (c + m * yp1)
    anp1_new <- alpha * an + beta * anp1
    bnp1_new <- alpha * bn + beta * bnp1
    scale <- inv_pow2(bnp1_new)
    list(an = anp1 * scale, anp1 = anp1_new * scale, bn = bnp1 * scale, bnp1 = bnp1_new * scale)
  }
  # Group steps so division by the current denominator happens only once
  # per group. This reduced buffer writes in the XLA-generated code examined;
  # the benefit depends on the compiler and backend.
  n_unroll <- 4L
  k0 <- ((a + 1) * u) * u
  r0 <- c1 / c
  # Divide by the current denominator so it becomes 1 and the numerator
  # becomes the approximation r. Since r is positive, we can store it as
  # negative to mark an element as finished, then use abs(r) in calculations.
  # This avoids a separate completion flag. XLA was found to duplicate this
  # loop's computation for each value carried into the next iteration, so
  # carrying fewer values reduced that work.
  frac <- nv_while(
    init = list(i = nv_scalar(0L), an = nv_fill_like(lambda, 0), bn = r0 / k0, r = nv_ifelse(use_asym, -r0, r0)),
    cond = function(i, an, bn, r) {
      nv_any(r > 0) & (i < 1000L)
    },
    body = function(i, an, bn, r) {
      done <- r < 0
      r <- nv_abs(r)
      cur <- list(an = an, anp1 = r, bn = bn, bnp1 = nv_fill_like(r, 1))
      # The numerator and denominator at the step that converged, held for
      # elements that converge during this trip
      fa <- r
      fb <- cur$bnp1
      conv <- done
      for (j in seq_len(n_unroll)) {
        m <- nv_convert(i, float_dtype) + j
        nxt <- step(m, cur$an, cur$anp1, cur$bn, cur$bnp1)
        # |A' / B' - A / B| <= eps A' / B', without dividing
        now <- !conv & (nv_abs(nxt$anp1 * cur$bnp1 - cur$anp1 * nxt$bnp1) <= eps * nv_abs(nxt$anp1 * cur$bnp1))
        fa <- nv_ifelse(now, nxt$anp1, fa)
        fb <- nv_ifelse(now, nxt$bnp1, fb)
        conv <- conv | now
        cur <- nxt
      }
      inv_b <- 1 / cur$bnp1
      r_new <- nv_ifelse(done, r, nv_ifelse(conv, fa / fb, cur$anp1 * inv_b))
      list(
        i = i + n_unroll,
        an = cur$an * inv_b,
        bn = cur$bn * inv_b,
        r = nv_ifelse(conv, -r_new, r_new)
      )
    }
  )

  # The fraction's leading factor is x^a (1 - x)^b / B(a, b), which equals
  # (n - k) * p * P(X = k) in either orientation. Combine it with the fraction
  # on the log scale to avoid multiplying tiny probabilities. A converged
  # fraction is stored negated, so -frac$r recovers its positive value.
  log_frac <- nv_log(n - k) + nv_log(p) + binom_log_density(k, n, p, n_terms) + nv_log(-frac$r)
  log_direct <- nv_ifelse(use_asym, log_asym, nv_ifelse(frac$r < 0, log_frac, NaN))
  # The direct tail is P(X > k) where `upper`, and P(X <= k) otherwise
  direct_wanted <- if (lower_tail) !upper else upper
  if (log_p) {
    # Rounding can take the direct tail a hair above 1
    complement <- log1mexp(nv_ifelse(direct_wanted, -1, nv_pmin(log_direct, 0)))
    nv_ifelse(direct_wanted, log_direct, complement)
  } else {
    direct <- nv_exp(log_direct)
    nv_ifelse(direct_wanted, direct, 1 - direct)
  }
}

#' @title The Binomial Distribution
#' @name nv_binomial
#' @description
#' Density (`nv_dbinom`), distribution function (`nv_pbinom`), and quantile
#' function (`nv_qbinom`) for the Binomial distribution with parameters `size`
#' and `prob`.
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`). These are counts, but held as floats.
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`. With `log_p = TRUE`, supply log-probabilities
#'   in \eqn{[-\infty, 0]} instead.
#' @param size ([`arrayish`])\cr
#'   Number of trials, zero or more. Either a scalar, or an array of exactly
#'   the shape of `x`/`q`/`p`.
#' @param prob ([`arrayish`])\cr
#'   Probability of success on each trial, in \eqn{[0, 1]}, shaped like
#'   `size`.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the densities/probabilities are given as logarithms. For
#'   `nv_qbinom` this describes the input `p`.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le x)}; otherwise,
#'   \eqn{P(X > x)}.
#' @details
#' The Binomial distribution with `size` \eqn{= n} and `prob` \eqn{= p} has
#' probability mass function:
#' \deqn{p(x) = \binom{n}{x} p^x (1-p)^{n-x}, \quad x = 0, \ldots, n}
#' and zero elsewhere.
#'
#' The functions compute at `f32` or `f64`, the data type of `x`/`q`/`p`.
#'
#' Invalid arguments give `NaN`, as in base R, but without a warning. That is
#' `prob` outside \eqn{[0, 1]}, or a negative `size`. A non-whole `size` is
#' also invalid for `nv_dbinom` and `nv_pbinom`, while `nv_qbinom` rounds it.
#' An infinite `size` is invalid for `nv_pbinom` and `nv_qbinom`.
#'
#' `nv_dbinom` is zero at a non-whole `x`. `nv_pbinom` rounds `q` down to a
#' whole number, after adding `1e-7` so that a `q` just below a whole number
#' counts as it.
#'
#' `nv_dbinom` uses the saddle point expansion of
#' `r cite_bib("loader2000fast")`, as base R does. `nv_pbinom` uses the
#' incomplete beta function methods of `r cite_bib("didonato1992algorithm")`
#' that base R's [stats::pbeta()] uses: a continued fraction, and an asymptotic
#' expansion for large `size` near the mean. If the continued fraction has not
#' converged after 1000 iterations, the result is `NaN`.
#'
#' `nv_qbinom` finds the smallest \eqn{x} with \eqn{P(X \le x) \ge p}, starting
#' from a Cornish-Fisher approximation as base R does. Like base R, it first
#' lowers `p` by a relative tolerance of \eqn{8\epsilon} (\eqn{2\epsilon} on
#' the log scale), where \eqn{\epsilon} is the machine epsilon of the data
#' type, so that a `p` computed by `nv_pbinom` maps back to its quantile. At
#' `f32` the tolerance is correspondingly coarser. The mapping back is not
#' guaranteed: it fails where neighbouring probabilities round to the same
#' value, such as close to zero or one, and the search evaluates the
#' distribution function afresh, which can differ from `nv_pbinom`'s value in
#' the last few units in the last place. Such a `p` can then map to the
#' neighbouring quantile; this is most likely on the log scale and at `f32`.
#'
#' @section Accuracy:
#' The figures here are relative errors of the returned value measured in our
#' tests, not guaranteed bounds. With `log = TRUE` or `log_p = TRUE` they are
#' relative errors of the log-density or log-probability itself.
#'
#' At `f64`, against high-precision (MPFR) references and base R for `size` up
#' to \eqn{10^{13}} (\eqn{10^{15}} for `nv_pbinom`), the relative error was
#' around \eqn{10^{-14}} or less. It grows in proportion to the magnitude of
#' the logarithm of the density or tail probability involved, to about
#' \eqn{2 \times 10^{-13}}{2e-13} for those near the bottom of the `f64` range
#' (around \eqn{10^{-300}}).
#'
#' At `f32`, against base R evaluated on the same `f32` inputs for `size` up to
#' \eqn{10^7}, the relative error was below \eqn{10^{-5}}, except for
#' densities and tail probabilities below about \eqn{10^{-14}}, where it
#' reached about \eqn{4 \times 10^{-5}}{4e-5}.
#'
#' `f32` holds every whole number only up to \eqn{2^{24}}, and `f64` up to
#' \eqn{2^{53}}. XLA (at least on the CPU backend) treats subnormal numbers
#' as zero, so a subnormal `prob` acts as `prob = 0`: `nv_dbinom(1, 2, 1e-310)`
#' at `f64` is zero, where base R gives `2e-310`.
#'
#' @section Gradients:
#' `nv_dbinom` can be differentiated with respect to `prob`, including at
#' `prob` of 0 and 1. Where the density is zero there, as at `prob = 0` for
#' `x > 0`, the log-density is `-Inf` and has no derivative: it diverges. With
#' `log = TRUE` the gradient returned there is zero, which is a convention of
#' this implementation, not the mathematical derivative.
#' `nv_pbinom` and `nv_qbinom` iterate until convergence in a
#' [nv_while()] loop, so [gradient()] cannot differentiate them.
#'
#' @templateVar dist binom
#' @templateVar params `size` or `prob`
#' @template section_distribution_dtype
#' @references
#' `r format_bib("loader2000fast", "didonato1992algorithm", "dekker1971floating")`
#' @seealso [nv_rbinom()] for sampling from a Binomial distribution.
#' @return ([`arrayish`])\cr
#' `nv_dbinom()`, `nv_pbinom()`, and `nv_qbinom()` return an [`arrayish`] with
#' the shape and data type of `x`/`q`/`p`.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 1, 2, 5, 10))
#' nv_dbinom(x, size = 10, prob = 0.3)
#' nv_dbinom(x, size = 10, prob = 0.3, log = TRUE)
#'
#' nv_pbinom(x, size = 10, prob = 0.3)
#' nv_pbinom(x, size = 10, prob = 0.3, lower_tail = FALSE)
#' nv_pbinom(x, size = 10, prob = 0.3, log_p = TRUE)
#'
#' p <- nv_array(c(0.025, 0.5, 0.975))
#' nv_qbinom(p, size = 10, prob = 0.3)
#' nv_qbinom(p, size = 10, prob = 0.3, lower_tail = FALSE)
#' nv_qbinom(nv_array(c(-700, -2, -0.1), dtype = "f64"), size = 1000, prob = 0.3, log_p = TRUE)
NULL

#' @rdname nv_binomial
#' @export
nv_dbinom <- jit(
  function(x, size, prob, log = FALSE) {
    assert_flag(log)
    args <- promote_distribution_args(x = x, size = size, prob = prob)
    x <- args$x
    size <- args$size
    prob <- args$prob
    op_dtype <- assert_rng_float_dtype(dtype(x), arg = "x")

    # Match base R rules for returning NaN. Range checks do not detect NaN, so
    # tested directly. Keep parameter checks independent of x to avoid expanding
    # scalar parameters to x's shape.
    invalid_param <- nv_is_nan(size) |
      nv_is_nan(prob) |
      (prob < 0) |
      (prob > 1) |
      (size < 0) |
      is_nonint(size)
    invalid <- invalid_param | nv_is_nan(x)
    # Match base R by rounding after checks. Values of x outside the whole-number
    # tolerance in `is_nonint()` are mapped to -1, giving probability zero.
    # Invalid elements get safe values to avoid gradient poisoning.
    x_whole <- nv_ifelse(nv_is_nan(x), 0, nv_ifelse(is_nonint(x), -1, nv_round(x)))
    size <- nv_ifelse(invalid_param, 1, nv_round(size))
    prob <- nv_ifelse(invalid_param, 0.5, prob)

    # Choose series length for Loader's saddle point expansion (`binom_bd0()`)
    # and compute log-density
    n_terms <- if (op_dtype == "f32") 4L else 8L
    log_density <- binom_log_density(x_whole, size, prob, n_terms)
    # Resolve based on log
    density <- if (log) {
      log_density
    } else {
      # Next to an end of the support, so at x = 1 for prob = 0 and x = size - 1
      # for prob = 1, the density is zero but gradient is not. However, just
      # exp(log_density) loses gradient, so compute directly for these cases.
      finite_size <- nv_is_finite(size)
      edge_low <- (prob == 0) & (x_whole == 1) & finite_size
      edge_high <- (prob == 1) & (x_whole == size - 1) & finite_size

      # Keep safety checks and direct calcs independent of `x`, so that scalar
      # `size` and `prob` produce scalar results until final selection.

      # Note: always 0/1, but needed to keep grandient flow.
      p_low <- nv_ifelse(prob == 0, prob, 0)
      p_high <- nv_ifelse(prob == 1, prob, 1)
      # Safe guard size for infinite inputs
      size_safe <- nv_ifelse(finite_size, size, 2)
      nv_ifelse(
        edge_low,
        # n p (1 - p)^{n-1}
        size_safe * p_low * nv_exp((size_safe - 1) * nv_log1p(-p_low)),
        nv_ifelse(
          edge_high,
          # n (1 - p) p^{n-1}
          size_safe * (1 - p_high) * nv_exp((size_safe - 1) * nv_log(p_high)),
          # easy cases don't need direct computation
          nv_exp(log_density)
        )
      )
    }
    nv_ifelse(invalid, NaN, density)
  },
  static = "log"
)

#' @rdname nv_binomial
#' @export
nv_pbinom <- jit(
  function(q, size, prob, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(q = q, size = size, prob = prob)
    q <- args$q
    size <- args$size
    prob <- args$prob
    # One convergence tolerance per width -- see `binom_cdf()`
    op_dtype <- assert_rng_float_dtype(dtype(q), arg = "q")

    # Resolve to NaN matching base R rules. The parameters' own conditions are
    # kept apart from q's, so that a scalar `size` or `prob` stays scalar.
    invalid_param <- nv_is_nan(prob) |
      !nv_is_finite(size) |
      is_nonint(size) |
      (size < 0) |
      (prob < 0) |
      (prob > 1)
    invalid <- invalid_param | nv_is_nan(q)
    size <- nv_round(size)
    # base R's fuzz, so that a q a hair below a whole number counts as it. At
    # f32, q + 1e-7 rounds up to the next whole number from as far as half an
    # ulp below it, so the gap to it is tested instead: (k + 1) - q is exact
    # wherever it can be small.
    k <- if (op_dtype == "f32") {
      k_floor <- nv_floor(q)
      nv_ifelse((k_floor + 1) - q <= 1e-7, k_floor + 1, k_floor)
    } else {
      nv_floor(q + 1e-7)
    }

    # Below the support, at or above `size`, and at `prob` of 0 or 1 the
    # distribution function is resolved directly, in base R's order. Elsewhere
    # 0 <= k < size and 0 < prob < 1, and the stand-ins keep the computation
    # well defined on the resolved elements.
    below <- q < 0
    above <- !below & (k >= size)
    is_zero <- below | (!above & (prob == 1))
    is_one <- above | (!below & (prob == 0))
    resolved <- invalid | is_zero | is_one
    # Stand-ins for the parameters are chosen on their own conditions, and the
    # stand-in k = 0 lies in [0, size) for any of them
    k_safe <- nv_ifelse(resolved, 0, k)
    size_safe <- nv_ifelse(invalid_param | (size < 1), 1, size)
    prob_safe <- nv_ifelse(invalid_param | (prob == 0) | (prob == 1), 0.5, prob)
    cdf <- binom_cdf(k_safe, size_safe, prob_safe, lower_tail, log_p, op_dtype)

    # The values of P(X <= k) of 0 and 1, in the scale and tail asked for
    if (lower_tail) {
      at_zero <- if (log_p) -Inf else 0
      at_one <- if (log_p) 0 else 1
    } else {
      at_zero <- if (log_p) 0 else 1
      at_one <- if (log_p) -Inf else 0
    }
    nv_ifelse(
      invalid,
      NaN,
      nv_ifelse(is_zero, at_zero, nv_ifelse(is_one, at_one, cdf))
    )
  },
  static = c("lower_tail", "log_p")
)

#' @rdname nv_binomial
#' @export
nv_qbinom <- jit(
  function(p, size, prob, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(p = p, size = size, prob = prob)
    p <- args$p
    size <- args$size
    prob <- args$prob
    op_dtype <- assert_rng_float_dtype(dtype(p), arg = "p")
    eps <- if (op_dtype == "f32") 2^-23 else 2^-52

    # base R rounds `size` here without first checking it is whole
    size <- nv_round(size)
    # Valid range checks for p, and resolution of its endpoints
    if (log_p) {
      in_range <- p <= 0
      at_bottom <- p == -Inf
      at_top <- p == 0
    } else {
      in_range <- (p >= 0) & (p <= 1)
      at_bottom <- p == 0
      at_top <- p == 1
    }
    # Resolve to NaN matching base R rules
    invalid <- !in_range |
      nv_is_nan(p) |
      !nv_is_finite(size) |
      !nv_is_finite(prob) |
      (size < 0) |
      (prob < 0) |
      (prob > 1)
    # `at_bottom`/`at_top` refer to P(X <= x) at the support's ends
    if (!lower_tail) {
      swap <- at_bottom
      at_bottom <- at_top
      at_top <- swap
    }
    # base R resolves the ends of `p` first, then degenerate distributions
    is_zero <- at_bottom | (!at_top & ((prob == 0) | (size == 0)))
    is_size <- at_top | (prob == 1)
    resolved <- invalid | is_zero | is_size
    size_safe <- nv_ifelse(resolved, 1, size)
    prob_safe <- nv_ifelse(resolved, 0.5, prob)
    p_safe <- nv_ifelse(resolved, if (log_p) -1 else 0.5, p)

    # base R's fuzz, so that the search is continuous from the left and maps
    # the p of a support point back to it despite rounding
    target <- if (log_p) {
      fuzzed <- p_safe * (if (lower_tail) 1 + 2 * eps else 1 - 2 * eps)
      nv_ifelse(nv_is_finite(fuzzed), fuzzed, p_safe)
    } else if (lower_tail) {
      # Just above the smallest normal number the fuzz makes p subnormal, which
      # XLA flushes to zero; p is left as it is there
      fuzzed <- p_safe * (1 - 8 * eps)
      nv_ifelse(fuzzed > 0, fuzzed, p_safe)
    } else {
      nv_ifelse(1 - p_safe > 32 * eps, p_safe * (1 + 8 * eps), p_safe)
    }
    # Whether the support point `k` is at or above the quantile; this is
    # monotone in `k`, false below the quantile and true from it on
    # A NaN from a failed evaluation of the distribution function is reported
    # as such, rather than read as either answer
    at_or_above <- function(k) {
      cdf <- binom_cdf(k, size_safe, prob_safe, lower_tail, log_p, op_dtype)
      list(above = if (lower_tail) cdf >= target else cdf < target, failed = nv_is_nan(cdf))
    }

    # Cornish-Fisher approximation to the quantile, as base R starts from
    q_safe <- 1 - prob_safe
    sigma <- nv_sqrt(size_safe * prob_safe * q_safe)
    gamma <- (q_safe - prob_safe) / sigma
    # z is clamped to [-40, 40], beyond any probability on the plain scale: at
    # an extreme log probability z * z overflows, or nv_qnorm() itself returns
    # NaN (below log p = -8.9e307), either of which would make the start NaN
    # and stop the search where it began. A NaN z only arises from a vanishing
    # probability in the tail asked for, so it takes that tail's end. Out there
    # the start is only a guess, and the search walks from it.
    z <- nv_qnorm(p_safe, lower_tail = lower_tail, log_p = log_p)
    z <- nv_ifelse(nv_is_nan(z), if (lower_tail) -40 else 40, nv_pmin(nv_pmax(z, -40), 40))
    start <- nv_round(size_safe * prob_safe + sigma * (z + gamma * (z * z - 1) / 6))

    # The quantile lies in (lo, hi]: `lo` is below it, `hi` at or above it.
    # From the start, step away in the direction it points, doubling the step
    # until the quantile is bracketed, and then bisect. Each probe lies strictly
    # inside (lo, hi). A `size` too large for its whole numbers to all be held
    # can leave no such probe, which ends the search too.
    probe_at <- function(lo, hi, step, started, galloping, down) {
      probe <- nv_ifelse(
        !started,
        start,
        nv_ifelse(
          galloping,
          nv_ifelse(down, hi - step, lo + step),
          nv_floor((lo + hi) / 2)
        )
      )
      nv_pmin(nv_pmax(probe, lo + 1), hi - 1)
    }
    is_active <- function(lo, hi, probe) (hi - lo > 1) & (probe > lo) & (probe < hi)
    like <- target + size_safe + prob_safe
    no <- nv_fill_like(like, 0) > 1
    search <- nv_while(
      init = list(
        i = nv_scalar(0L),
        lo = nv_fill_like(like, -1),
        hi = nv_fill_like(like, 0) + size_safe,
        step = nv_fill_like(like, 1),
        started = no,
        galloping = !no,
        down = no
      ),
      cond = function(i, lo, hi, step, started, galloping, down) {
        probe <- probe_at(lo, hi, step, started, galloping, down)
        nv_any(is_active(lo, hi, probe)) & (i < 2200L)
      },
      body = function(i, lo, hi, step, started, galloping, down) {
        probe <- probe_at(lo, hi, step, started, galloping, down)
        active <- is_active(lo, hi, probe)
        answer <- at_or_above(nv_ifelse(active, probe, 0))
        above <- answer$above
        # A failed evaluation ends that element's search with a NaN bracket,
        # and so a NaN result
        failed <- active & answer$failed
        # Galloping goes on while each probe lands on the same side as the last
        onward <- nv_ifelse(down, above, !above)
        list(
          i = i + 1L,
          lo = nv_ifelse(failed, NaN, nv_ifelse(active & !above, probe, lo)),
          hi = nv_ifelse(failed, NaN, nv_ifelse(active & above, probe, hi)),
          step = nv_ifelse(started & galloping & onward, 2 * step, step),
          started = started | active,
          galloping = galloping & (!started | onward),
          down = nv_ifelse(started, down, above)
        )
      }
    )

    nv_ifelse(
      invalid,
      NaN,
      nv_ifelse(is_zero, 0, nv_ifelse(is_size, size, search$hi))
    )
  },
  static = c("lower_tail", "log_p")
)
