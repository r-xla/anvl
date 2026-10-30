#' @include jit.R
NULL

## Probability distributions ---------------------------------------------------

# The operand -- `x`/`q`/`p`, always the first argument -- governs the data type
# the distribution is evaluated at; the parameters follow it. An integer operand
# has no float precision to govern with, so it is first converted to the widest
# float among the parameters that are arrays, or to the default float if none
# is. R values do not count: they have no data type of their own. The numeric
# check runs before the promotion, so a boolean operand is reported as itself
# rather than as a failure to bring a parameter to its data type.
promote_distribution_args <- function(...) {
  args <- list(...)
  operand <- names(args)[[1L]]
  operand_dtype <- assert_numeric_dtype(
    peek_dtype(args[[1L]]),
    arg = operand,
    hint = "Convert it with `nv_convert()`."
  )
  if (!is_dtype_float(operand_dtype)) {
    args[[1L]] <- nv_convert(args[[1L]], distribution_float(args[-1L]))
  }
  do.call(as_anvl_arrays, c(args, list(.promote = promotion_like(operand))))
}

# The float an integer operand is evaluated at: the widest float among the
# parameter arrays, or the default float.
distribution_float <- function(params) {
  avals <- lapply(params, to_abstract)
  dtypes <- lapply(avals[!vapply(avals, is_rdata, logical(1L))], function(aval) aval$dtype)
  floats <- Filter(is_dtype_float, dtypes)
  if (length(floats) == 0L) {
    return(default_float())
  }
  floats[[which.max(vapply(floats, dtype_width, integer(1L)))]]
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
#' shape of `x`/`q`/`p` and the float data type described under Data Types.
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
#' same shape as `x`/`q`/`p` and the float data type described under Data Types.
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

    # The probability below and above `q`
    p_below <- function() resolve_ends(1, 0, (q_int - min) / width)
    p_above <- function() resolve_ends(0, 1, (max - q_int) / width)

    u <- if (lower_tail) p_below() else p_above()

    # Reversed/non-finite interval is NaN to match base R: `valid` flag to track
    valid <- nv_is_finite(min) & nv_is_finite(max) & (max >= min)

    if (!log_p) {
      return(nv_ifelse(valid, u, NaN))
    }

    # To maintain accuracy of log near 1, switch to log1p in opposite tail mid way
    v <- if (lower_tail) p_above() else p_below()
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
#' float data type described under Data Types and the shape after scalar
#' broadcasting with `rate`.
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

## Discrete distribution helpers -----------------------------------------------

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

# Returns x unchanged, with no gradient flowing back through it (anvl currently
# has no stop gradient primitive).
#
# Used where the derivative along this path should be zero, or where f(x) should
# be computed by one formula but differentiated by another, because autodiff
# through f is inaccurate or overflows. Compute the value on a detached input,
# f(stop_gradient(x)), and if there is some g(x) that has the same derivative
# as f, attach it with
#
#   f(stop_gradient(x)) + (g(x) - stop_gradient(g(x)))
#
# The added term is zero, but its derivatives is g's (hence f's), to every
# order. So only g's derivative is used, never its value. For example,
# `log_ratio()` uses:
#
#   f = log(x / y)         accurate value, but derivative can overflow
#   g = log(x) - log(y)    inaccurate value when x is near y, clean derivative
#
# Only if there is no such g, attach f's derivative at the detached input:
#
#   f(stop_gradient(x)) + (x - stop_gradient(x)) * f'(stop_gradient(x))
#
# This gives the right first derivative, _but_ second and higher derivatives
# come out as zero. Note that f' must be passed the detached input.
stop_gradient <- function(x) {
  float_dtype <- dtype(x)
  int_dtype <- if (float_dtype == "f32") "i32" else "i64"
  nv_bitcast_convert(nv_bitcast_convert(x, int_dtype), float_dtype)
}

# Returns a power of two, s, such that 1 <= x * s < 2.
#
# Used to rescale values in loops before they overflow or underflow.
# Scaling by a power of two preserves the significand, introduces no rounding
# if the scaled values remain in the normal range.
#
# Requires positive, normal, finite x, with a scaling factor s that is
# also normal and finite.
inv_pow2 <- function(x) {
  is_f32 <- dtype(x) == "f32"
  bits_dtype <- if (is_f32) "ui32" else "ui64"
  n_mantissa <- if (is_f32) 23L else 52L
  exponent_max <- if (is_f32) 254L else 2046L
  e <- nv_shift_right_logical(nv_bitcast_convert(x, bits_dtype), n_mantissa)
  nv_bitcast_convert(nv_shift_left(exponent_max - e, n_mantissa), dtype(x))
}

# Evaluates and returns compute() when the scalar boolean pending is TRUE;
# otherwise returns init without evaluating compute().
#
# Skips expensive work when no element in a batch needs it. For example:
#
#   out <- skip_unless(nv_any(mask), nv_fill_like(x, 0), function() {
#     expensive(nv_ifelse(mask, x, safe_value))
#   })
#   nv_ifelse(mask, out, cheap)
#
# Here, expensive() runs only if at least one element needs it. Elements
# outside mask receive safe_value to keep the batched computation valid.
#
# Uses a loop that runs at most once because nv_if() executes both branches
# on XLA's CPU backend. This helper does not support differentiation through
# the loop.
skip_unless <- function(pending, init, compute) {
  nv_while(
    init = list(pending = pending, value = init),
    cond = function(pending, value) pending,
    body = function(pending, value) list(pending = nv_scalar(FALSE), value = compute())
  )$value
}

# Computes log(x / y) for positive normal x and y, rescaling y by a power
# of two to avoid overflow or loss of precision in the ratio.
# Attaches the derivatives 1 / x and -1 / y explicitly: in reverse mode,
# XLA otherwise rewrites the derivative with respect to x as y / (x * y),
# whose intermediate product can overflow or underflow.
# Each gradient correction is zero unless its input is positive, normal,
# and finite; the corrections leave the computed value unchanged.
log_ratio <- function(x, y) {
  x_value <- stop_gradient(x)
  y_value <- stop_gradient(y)
  is_f32 <- dtype(x) == "f32"
  k <- if (is_f32) 126 else 1022
  r <- x_value / y_value
  over <- r == Inf
  under <- r < 2^-k
  y_scaled <- nv_ifelse(over, y_value * 2^k, nv_ifelse(under, y_value * 2^-k, y_value))
  # Builds the shift with x's data type; choosing between plain R values
  # would instead materialize it with the default floating-point type.
  shift <- nv_scalar_like(x, k * base::log(2))
  value <- nv_log(x_value / y_scaled) + nv_ifelse(over, shift, nv_ifelse(under, -shift, 0))

  # Returns zero with the derivative of log(v) for positive, normal, finite v,
  # taken from log's own rule so that it is right to every order: dividing by
  # the detached v would give the right first derivative but zero second.
  # Otherwise returns zero without a gradient correction. Using 1 as the input
  # in that case also prevents NaN gradients in the untaken branch.
  smallest_normal <- if (is_f32) 2^-126 else 2^-1022
  log_derivative <- function(v, v_value) {
    normal <- (v_value >= smallest_normal) & (v_value != Inf)
    log_v <- nv_log(nv_ifelse(normal, v, 1))
    nv_ifelse(normal, log_v - stop_gradient(log_v), 0)
  }
  value + log_derivative(x, x_value) - log_derivative(y, y_value)
}

# Error of Stirling's approximation to log(n!),
#   stirlerr(n) = log(n!) - log(sqrt(2 * pi * n) * (n / e)^n),
# for whole n >= 1, same approach as R's `stirlerr()`: exact values up to
# n = 15, and beyond an asymptotic series 1/(12 n) - 1/(360 n^3) + .... From
# n = 16 the series has converged to double precision by its seventh term (the
# term count R uses), and to single precision by its fourth. The exact values
# are gathered from a table, which benched faster than picking them out with a
# chain of 15 selects (by 18% for `nv_dbinom()` in f64, and ~5% for `nv_dpois()`
# in f32, at 1.5e6 elements).
#
# Only whole n are handled!!
# NOTE: Revisit before reusing this for other distributions: Student's t needs
# half-integer (and, for non-whole degrees of freedom, general real) arguments.
stirlerr <- function(n) {
  # Gather only takes a vector of indices, so flatten and restore the shape
  table <- nv_array(stirlerr_table, dtype = dtype(n))
  idx <- nv_reshape(nv_convert(nv_clamp(n, 1, 15), "i32"), -1L)
  tabled <- nv_reshape(table[idx], shape(n))
  n_series <- nv_pmax(n, 16)
  coefs <- if (dtype(n) == "f32") stirlerr_series[4:7] else stirlerr_series
  series <- horner(1 / (n_series * n_series), coefs) / n_series
  nv_ifelse(n <= 15, tabled, series)
}

# 1/12 - 1/(360 n^2) + 1/(1260 n^4) - ..., as a polynomial in 1 / n^2, highest
# power first. See `src/nmath/stirlerr.c` in R's sources for the values in
# `stirlerr_series` and `stirlerr_table`.
stirlerr_series <- c(1 / 156, -691 / 360360, 1 / 1188, -1 / 1680, 1 / 1260, -1 / 360, 1 / 12)

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

# Deviance term bd0(x, np) = x log(x / np) + np - x of Loader's saddle point
# expansion, for x >= 1 and np > 0 (Loader, 2000; R's `bd0()`). Near x = np the
# direct form cancels catastrophically, so it is replaced there by the series
#   d v + 2 x v (v^2 / 3 + v^4 / 5 + ...),  d = x - np,  v = d / (x + np),
# truncated where it is accurate to working precision for |v| < 0.1. The
# coefficients 1 / (2j + 3) are highest power of v^2 first.
#
# A caller that knows x - np more accurately than x minus the rounded np should
# pass `d` (eg the binomial forms it from n and p with `sub_exact_prod()`). Near
# the mean the derivative of bd0 is proportional to d, so with the rounded
# difference it would be lost wherever np rounds to x.
bd0_f64_coefs <- 1 / c(17, 15, 13, 11, 9, 7, 5, 3)
bd0_f32_coefs <- 1 / c(9, 7, 5, 3)

bd0 <- function(x, np, d = x - np) {
  is_f32 <- dtype(x) == "f32"
  # v and the test for |v| < 0.1 are scale-free, so near the top of the range
  # x, np and d are scaled down first: x + np could overflow, and the
  # reciprocal XLA divides by could be subnormal. The scaling is selected rather
  # than factored out, which XLA would undo by rewriting c x + c np as
  # c (x + np).
  huge <- nv_pmax(x, np) > (if (is_f32) 2^120 else 2^1016)
  xs <- nv_ifelse(huge, x * 2^-4, x)
  nps <- nv_ifelse(huge, np * 2^-4, np)
  ds <- nv_ifelse(huge, d * 2^-4, d)
  ss <- xs + nps
  near <- nv_abs(ds) < 0.1 * ss
  v <- nv_ifelse(near, ds / ss, 0)
  w <- v * v
  series <- d * v + x * v * w * (2 * horner(w, if (is_f32) bd0_f32_coefs else bd0_f64_coefs))
  # np - x first: near the top of the range x log(x / np) + np can overflow
  # where the whole cannot
  direct <- (np - x) - x * log_ratio(np, x)
  nv_ifelse(near, series, direct)
}

# log(1 + t) - t, for small |t|, by the series in r = t / (2 + t) of R's
# `log1pmx()` and `rlog1()`. As log(1 + t) = 2 atanh(r) = 2 (r + r^3/3 + r^5/5 + ...)
# and 2 r - t = -r t,
#   log(1 + t) - t = r (2 r^2 S(r^2) - t),  S(y) = 1/3 + y/5 + y^2/7 + ...
# Subtracting t from log1p(t) directly loses precision near zero. The caller
# chooses how many terms of S its range of t needs.
log1pmx <- function(t, n_terms) {
  r <- t / (2 + t)
  y <- r * r
  # S(y) to n_terms terms: coefficients 1 / (2k + 1), k = n_terms, ..., 1,
  # highest power first
  r * (2 * y * horner(y, 1 / (2 * rev(seq_len(n_terms)) + 1)) - t)
}

# Rounds q down, treating values within 1e-7 below an integer as that integer.
# For f64, uses floor(q + 1e-7), as base R does. For f32, adding the tolerance
# can itself round q up to the next integer, even outside that tolerance.
# Instead, compares the distance to the next integer directly.
whole_count <- function(q) {
  if (dtype(q) == "f32") {
    k <- nv_floor(q)
    nv_ifelse((k + 1) - q <= 1e-7, k + 1, k)
  } else {
    nv_floor(q + 1e-7)
  }
}

# The values a probability of 0 (dt_zero) or 1 (dt_one) takes once converted
# to the requested tail and scale: with lower_tail = FALSE, p becomes 1 - p,
# and with log_p = TRUE, it becomes log(p). These correspond to base R's R_DT_0
# and R_DT_1.
dt_zero <- function(lower_tail, log_p) {
  if (lower_tail) (if (log_p) -Inf else 0) else (if (log_p) 0 else 1)
}
dt_one <- function(lower_tail, log_p) dt_zero(!lower_tail, log_p)

# Adjusts p by a few ulps before a discrete quantile search, as base R does.
# Rounding can leave the CDF at the correct count just below the target,
# causing the search to return the next count. To compensate, move the
# target slightly down for the lower tail or up for the upper tail.
# On the log scale, p is negative, so these adjustments use factors
# greater than or less than one, respectively.
# Leave p unchanged if the adjustment would underflow to zero, become
# non-finite, or affect an upper-tail probability too close to one.
quantile_target <- function(p, lower_tail, log_p) {
  eps <- if (dtype(p) == "f32") 2^-23 else 2^-52
  if (log_p) {
    fuzzed <- p * (if (lower_tail) 1 + 2 * eps else 1 - 2 * eps)
    nv_ifelse(nv_is_finite(fuzzed), fuzzed, p)
  } else if (lower_tail) {
    fuzzed <- p * (1 - 8 * eps)
    nv_ifelse(fuzzed > 0, fuzzed, p)
  } else {
    nv_ifelse(1 - p > 32 * eps, p * (1 + 8 * eps), p)
  }
}

# Returns a Cornish-Fisher quantile approximation using the mean, standard
# deviation and skewness. Used as a starting point for discrete quantile
# searches, as in base R.
# Extreme log probabilities can make nv_qnorm() return NaN or make z * z
# overflow. Clamp z to [-40, 40], replacing NaN with -40 for the lower tail
# or 40 for the upper tail.
# The subsequent search uses the original target probability, so clamping
# can increase the number of search steps but does not change the answer.
cornish_fisher <- function(p, mean, sd, skew, lower_tail, log_p) {
  z <- nv_qnorm(p, lower_tail = lower_tail, log_p = log_p)
  z <- nv_ifelse(nv_is_nan(z), if (lower_tail) -40 else 40, nv_pmin(nv_pmax(z, -40), 40))
  mean + sd * (z + skew * (z * z - 1) / 6)
}

# Finds the smallest whole y in [0, upper] satisfying a condition that
# remains true for all larger counts. Takes growing steps from start to
# bracket the answer, then bisects.
#
# satisfied(y) returns a list with sat = TRUE where the condition holds,
# and optionally failed = TRUE where evaluation failed. Failed elements
# return NaN, except at upper, which always counts as satisfied.
#
# Beyond the dtype's exact integer range, returns the smallest representable
# count satisfying the condition. A while loop has no gradient, so inputs
# must not carry one.
#
# Performance: batches run until the slowest element finishes. Probing start
# before the loop saves runtime overhead at the cost of extra compilation.
# Keep loop state small: XLA's CPU backend recomputes shared body work for
# each value carried out.
discrete_search <- function(start, satisfied, upper) {
  is_f32 <- dtype(start) == "f32"
  eps <- if (is_f32) 2^-23 else 2^-52
  check <- function(y) {
    s <- satisfied(y)
    list(sat = s$sat | (y == upper), failed = if (!is.null(s$failed)) s$failed & (y != upper))
  }
  # Bounds after probing `probe` where `active`; a failed evaluation sets both to
  # NaN, which stops that element's search
  narrow <- function(lo, hi, probe, active, s) {
    lo <- nv_ifelse(active & !s$sat, probe, lo)
    hi <- nv_ifelse(active & s$sat, probe, hi)
    if (!is.null(s$failed)) {
      failed <- active & s$failed
      lo <- nv_ifelse(failed, NaN, lo)
      hi <- nv_ifelse(failed, NaN, hi)
    }
    list(lo = lo, hi = hi)
  }

  start <- nv_ifelse(nv_is_nan(start), 0, nv_pmin(nv_pmax(start, 0), upper))
  # lo is unsatisfied (-1 until found); hi is satisfied (Inf until found).
  # Step lengths are ceiling(s), ceiling(2s), ceiling(4s), ..., s the initial step.
  # Below 2^51 in f64 (2^22 in f32) s is 1/2, so the lengths are 1, 1, 2, 4, ...
  # Repeating the first step avoids overshooting an answer two counts away.
  # Upward probes stop at upper; downward probes stop at zero.
  # s is at least one ulp of start, so the first step changes the probe.
  bounds <- narrow(-1, Inf, start, TRUE, check(start))
  step <- 0.5 * nv_pmax(nv_floor(start * (4 * eps)), 1)
  next_probe <- function(lo, hi, step) {
    up <- hi == Inf
    down <- !up & (lo == -1)
    move <- nv_ceiling(step)
    probe <- nv_ifelse(
      up,
      nv_pmin(lo + move, upper),
      nv_ifelse(down, nv_pmax(hi - move, 0), nv_floor(lo + 0.5 * (hi - lo)))
    )
    # NaN bounds fail both comparisons, so a failed element is never active
    list(probe = probe, active = (probe > lo) & (probe < hi), gallop = up | down)
  }
  # Enough trips to gallop from 0 to the largest float and bisect back
  max_trips <- if (is_f32) 320 else 2200
  out <- nv_while(
    init = list(n = nv_scalar_like(start, 0), lo = bounds$lo, hi = bounds$hi, step = step),
    cond = function(n, lo, hi, step) (n < max_trips) & nv_any(next_probe(lo, hi, step)$active),
    body = function(n, lo, hi, step) {
      nxt <- next_probe(lo, hi, step)
      bounds <- narrow(lo, hi, nxt$probe, nxt$active, check(nv_ifelse(nxt$active, nxt$probe, 0)))
      list(
        n = n + 1,
        lo = bounds$lo,
        hi = bounds$hi,
        step = nv_ifelse(nxt$active & nxt$gallop, 2 * step, step)
      )
    }
  )
  out$hi
}

## Binomial distribution helpers -----------------------------------------------

# Log of the Binomial probability mass at whole `x`, for whole `n >= 0` and
# `0 <= p <= 1`, as in R's `dbinom_raw()` (Loader's saddle point algorithm).
# Avoids constructing huge factorials or subtracting large, nearly equal
# log-factorial expressions.
# Each branch is fed safe stand-ins on the elements where it is not selected, so
# that it cannot poison the gradient. The stand-ins for `n` and `p` are chosen
# by conditions on `n` and `p` alone, so a scalar `n` or `p` stays scalar.
binom_log_density <- function(x, n, p) {
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
  lc <- stirlerr(n_int) -
    stirlerr(x_int) -
    stirlerr(n_int - x_int) -
    bd0(x_int, np, d) -
    bd0(n_int - x_int, nq, -d)
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

# Compute the scaled complementary error function exp(z^2) * erfc(z), z >= 0.
# For large z, the two factors overflow and underflow even though their product
# is finite. Above z_switch, use an asymptotic approximation to the product.
# The caller chooses the switch and term count together for the float type.
erfcx <- function(z, z_switch, n_terms) {
  use_series <- z > z_switch
  # Both branches are evaluated, so give each safe inputs where it is unused.
  z_direct <- nv_ifelse(use_series, 0, z)
  z_series <- nv_ifelse(use_series, z, z_switch)
  u <- 1 / (2 * z_series * z_series)
  # Evaluate 1 - u + 3 u^2 - 15 u^3 + ... by nested multiplication.
  # Dividing this by z * sqrt(pi) gives the asymptotic approximation.
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

# Compute the Stirling correction del(a) + del(b) - del(a + b), for a, b >= 8,
# as in R's `bcorr()`. Here del(x) is the error in the log-Gamma approximation:
# del(x) = log Gamma(x) - (x - 1/2) log(x) + x - log(2 pi) / 2.
# Combine the last two corrections algebraically rather than subtracting
# nearly equal values when one parameter is much larger than the other.
bcorr <- function(a0, b0) {
  # Put the smaller parameter first, so h <= 1. Then x = b / (a + b)
  # and c = 1 - x, calculated without subtracting x from 1.
  a <- nv_pmin(a0, b0)
  b <- nv_pmax(a0, b0)
  h <- a / b
  c <- h / (h + 1)
  x <- 1 / (h + 1)
  x2 <- x * x
  # Build s3, s5, ..., s11, where sj = 1 + x + ... + x^(j-1).
  # These let us write 1 - x^j as c * sj without cancellation.
  s <- Reduce(function(acc, i) c(acc, list(x + x2 * acc[[length(acc)]] + 1)), 1:4, init = list(x + x2 + 1))
  # Approximate del(t) by (c0 + c1/t^2 + c2/t^4 + ...) / t.
  # Store the coefficients in increasing order of the inverse powers.
  coefs <- c(
    0.0833333333333333,
    -0.00277777777760991,
    7.9365066682539e-4,
    -5.9520293135187e-4,
    8.37308034031215e-4,
    -0.00165322962780713
  )
  t <- 1 / (b * b)
  # Evaluate del(b) - del(a + b): weight each odd inverse power by its sj,
  # then multiply by c / b. Horner's method takes highest powers first.
  w <- horner(t, rev(c(list(coefs[[1L]]), Map(`*`, coefs[-1L], s))))
  w <- w * c / b
  # Add the correction for the smaller parameter, del(a).
  t <- 1 / (a * a)
  horner(t, rev(coefs)) / a + w
}

# Compute log I_x(a, b) using the asymptotic expansion of DiDonato and Morris
# (1992, Algorithm 708), as in R's `basym()`. The caller uses this for large
# a and b with 0 <= lambda = a - (a + b) x <= 0.03 * min(a, b).
# The beta argument x is represented by lambda rather than passed separately.
# Use a fixed expansion length for each float type; the R loops are unrolled
# during tracing, with no convergence loop at run time.
basym <- function(a, b, lambda, op_dtype) {
  is_f32 <- op_dtype == "f32"
  n_terms <- if (is_f32) 4L else 8L
  e0 <- 2 / base::sqrt(pi)
  e1 <- 2^-1.5

  # f measures departure from the beta mean, as a sum of terms
  # t - log(1 + t) = -log1pmx(t) with |t| <= 0.03, where r^2 < 2.5e-4 and 3
  # terms suffice for f32 and 6 for f64. The series avoids cancellation near
  # that mean; clamp small negative rounding errors before taking sqrt.
  n_log1pmx <- if (is_f32) 3L else 6L
  f <- nv_pmax(-(a * log1pmx(-lambda / a, n_log1pmx) + b * log1pmx(lambda / b, n_log1pmx)), 0)
  z0 <- nv_sqrt(f)
  z <- z0 / e1 * 0.5
  z2 <- f + f
  # Express the parameter balance using a ratio <= 1. r1 retains the sign
  # of the asymmetry, while w0 sets the scale of successive correction terms.
  a_lt_b <- a < b
  h <- nv_ifelse(a_lt_b, a / b, b / a)
  r0 <- 1 / (h + 1)
  r1 <- (b - a) / nv_pmax(a, b)
  w0 <- 1 / nv_sqrt(nv_pmin(a, b) * (h + 1))

  # Seed the coefficient recurrences from Algorithm 708. Keep its naming:
  # a0 supplies the expansion coefficients, and cc and dd are derived below.
  a0 <- list(r1 * 2 / 3)
  cc <- list(-a0[[1L]] / 2)
  dd <- list(a0[[1L]] / 2)
  # Start the two j recurrences with the scaled error function. Keeping the
  # exp(-f) factor out of the sum avoids underflow; it is restored in log form.
  j0 <- 0.5 / e0 * erfcx(z0, if (is_f32) 8 else 20, if (is_f32) 6L else 8L)
  j1 <- e1
  sum <- j0 + dd[[1L]] * w0 * j1

  s <- 1
  h2 <- h * h
  hn <- 1
  w <- w0
  znm1 <- z
  zn <- z2
  # Add corrections in pairs, at orders n and n + 1. Thus the final pair
  # includes order n_terms + 1, despite the loop ending at n_terms.
  for (n in seq(2L, n_terms, by = 2L)) {
    # Update the even and odd a0 coefficients using successive powers of h^2.
    hn <- hn * h2
    a0[[n]] <- r0 * 2 * (h * hn + 1) / (n + 2)
    s <- s + hn
    a0[[n + 1L]] <- r1 * 2 * s / (n + 3)
    # For each order, derive b0 and cc, then dd from the earlier coefficients.
    # These are the paper's finite recurrences, not convergence tests.
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
    # Advance both j recurrences and their powers of z, then weight the two
    # corrections by successive powers of w0.
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

  # Combine e0 * exp(-f) * exp(-bcorr(a, b)) * sum on the log scale.
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
  # at run time where no element needs it (see `skip_unless()`).
  log_asym <- skip_unless(nv_any(use_asym), nv_fill_like(lambda, 0), function() {
    basym(
      # This runs for the whole array if any element needs the expansion.
      # Give the other elements safe inputs within the expansion's domain.
      nv_ifelse(use_asym, a, 2000),
      nv_ifelse(use_asym, b, 2000),
      nv_ifelse(use_asym, lambda, 0),
      op_dtype
    )
  })

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

  # After each step, `inv_pow2()` gives the power of two that scales the
  # current denominator into [1, 2), without division. Scaling is exact while
  # the results remain normal; smaller values can still underflow. See the
  # comments above `step()` for the limits.
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
  log_frac <- nv_log(n - k) + nv_log(p) + binom_log_density(k, n, p) + nv_log(-frac$r)
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
#'   function (`q`). These are counts, given as integers or floats; see
#'   Non-Integer Values for values that are not whole numbers.
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
#' Invalid arguments give `NaN`, as in base R, but without a warning (that is
#' `prob` outside \eqn{[0, 1]}, or a `size` that is negative or not a whole
#' number). `size` is taken as a whole number with the same tolerance as `x`
#' (see Non-Integer Values).
#'
#' `nv_dbinom` uses the saddle point expansion of
#' `r cite_bib("loader2000fast")`, as base R does. `nv_pbinom` uses the
#' incomplete beta function methods of `r cite_bib("didonato1992algorithm")`
#' that base R's [stats::pbeta()] uses: a continued fraction, and an asymptotic
#' expansion for large `size` near the mean. If the continued fraction has not
#' converged after 1000 iterations, the result is `NaN`. `nv_qbinom` finds the
#' smallest \eqn{x} with \eqn{P(X \le x) \ge p}, starting from a Cornish-Fisher
#' approximation as base R does. Above `2^24` in `f32` and `2^53` in `f64` not
#' every whole number is a float, and `nv_qbinom` then returns the smallest float
#' \eqn{x} with \eqn{P(X \le x) \ge p}.
#'
#' @templateVar dist binom
#' @templateVar params `size` or `prob`
#' @template section_distribution_nonint
#' @section Gradients:
#' `nv_dbinom` can be differentiated with respect to `prob`, including at
#' `prob` of 0 and 1. Where the density is zero there, as at `prob = 0` for
#' `x > 0`, the log-density is `-Inf` and has no derivative: it diverges. With
#' `log = TRUE` the gradient returned there is zero, which is a convention of
#' this implementation, not the mathematical derivative.
#' `nv_pbinom` iterates until convergence in a [nv_while()] loop, so
#' [gradient()] cannot differentiate it. `nv_qbinom` is a step function, and
#' autodiff returns a zero gradient for all of its arguments.
#'
#' @template section_distribution_dtype
#' @references
#' `r format_bib("loader2000fast", "didonato1992algorithm", "dekker1971floating")`
#' @seealso [nv_rbinom()] for sampling from a Binomial distribution.
#' @return ([`arrayish`])\cr
#' `nv_dbinom()`, `nv_pbinom()`, and `nv_qbinom()` return an [`arrayish`] with
#' the shape of `x`/`q`/`p` and the float data type described under Data Types.
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
    # tolerance in `is_nonint()` are mapped to -1, giving probability zero, as
    # are negative values: the sign is tested before rounding, as base R does, so
    # a tiny negative `x` that rounds to zero is still outside the support.
    # Invalid elements get safe values to avoid gradient poisoning.
    x_whole <- nv_ifelse(nv_is_nan(x), 0, nv_ifelse(is_nonint(x) | (x < 0), -1, nv_round(x)))
    size <- nv_ifelse(invalid_param, 1, nv_round(size))
    prob <- nv_ifelse(invalid_param, 0.5, prob)

    log_density <- binom_log_density(x_whole, size, prob)
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
    op_dtype <- assert_rng_float_dtype(dtype(q), arg = "q")

    # Match base R rules for returning NaN, including non-finite size.
    # Keep parameter checks independent of q to avoid expanding scalar
    # parameters to q's shape. Check NaN directly where range checks miss it.
    invalid_param <- nv_is_nan(prob) |
      !nv_is_finite(size) |
      is_nonint(size) |
      (size < 0) |
      (prob < 0) |
      (prob > 1)
    invalid <- invalid_param | nv_is_nan(q)
    # Round size after checking it is within the whole-number tolerance.
    size <- nv_round(size)
    # Values just below a whole number count as that number, as in base R
    k <- whole_count(q)

    # Identify where the lower-tail probability P(X <= q) is exactly 0 or 1.
    # Check the support limits before prob = 0 or 1, matching base R's order.
    # Use the original q for the lower limit: even a tiny negative q is below
    # the support, regardless of whether the tolerance above rounds it to 0.
    below <- q < 0
    above <- !below & (k >= size)
    is_zero <- below | (!above & (prob == 1))
    is_one <- above | (!below & (prob == 0))
    resolved <- invalid | is_zero | is_one
    # binom_cdf() needs 0 <= k < size and 0 < prob < 1. Give invalid or
    # already-resolved elements safe inputs; their results are replaced below.
    # Keep parameter replacements independent of q so scalar parameters stay
    # scalar. The replacement k = 0 is valid for every size_safe >= 1.
    k_safe <- nv_ifelse(resolved, 0, k)
    size_safe <- nv_ifelse(invalid_param | (size < 1), 1, size)
    prob_safe <- nv_ifelse(invalid_param | (prob == 0) | (prob == 1), 0.5, prob)
    cdf <- binom_cdf(k_safe, size_safe, prob_safe, lower_tail, log_p, op_dtype)

    # Convert the known lower-tail probabilities 0 and 1 to the requested
    # tail and scale. Invalid inputs take precedence over them.
    nv_ifelse(
      invalid,
      NaN,
      nv_ifelse(is_zero, dt_zero(lower_tail, log_p), nv_ifelse(is_one, dt_one(lower_tail, log_p), cdf))
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
    op_dtype <- assert_rng_float_dtype(dtype(args$p), arg = "p")
    # A step function has a zero gradient, and the search below is a while
    # loop, which has none at all
    p <- stop_gradient(args$p)
    size <- stop_gradient(args$size)
    prob <- stop_gradient(args$prob)

    # Match base R by rounding size here, without requiring a whole-number input.
    # This deliberately differs from nv_dbinom/nv_pbinom, which return NaN for a
    # non-whole size: base R's qbinom applies R_forceint(n) with no R_nonint(n)
    # check, while dbinom and pbinom check it. Rounding happens before the
    # negative-size check, as in base R, so e.g. size = -0.4 is treated as 0.
    size <- nv_round(size)
    in_range <- if (log_p) p <= 0 else (p >= 0) & (p <= 1)
    at_bottom <- p == dt_zero(lower_tail, log_p)
    at_top <- p == dt_one(lower_tail, log_p)
    # Match base R rules for returning NaN. Both parameters must be finite;
    # p may be -Inf on the log scale, where it represents probability zero.
    invalid <- !in_range |
      !nv_is_finite(size) |
      !nv_is_finite(prob) |
      (size < 0) |
      (prob < 0) |
      (prob > 1)
    # Resolve probability endpoints before distributions concentrated at 0
    # or size, matching base R's order.
    is_zero <- at_bottom | (!at_top & ((prob == 0) | (size == 0)))
    is_size <- at_top | (prob == 1)
    resolved <- invalid | is_zero | is_size
    # Give invalid or already-resolved elements safe inputs for the search.
    # Their search results are replaced before returning.
    size_safe <- nv_ifelse(resolved, 1, size)
    prob_safe <- nv_ifelse(resolved, 0.5, prob)
    p_safe <- nv_ifelse(resolved, if (log_p) -1 else 0.5, p)

    target <- quantile_target(p_safe, lower_tail, log_p)
    # Whether k is at or above the quantile. The lower-tail CDF increases with
    # k; the upper tail decreases, and its comparison is strict. The search
    # takes k = size as satisfied without asking, so `binom_cdf()`, which needs
    # k < size, is given a safe count there. A failed CDF evaluation is
    # reported separately so a NaN cannot be mistaken for a search direction.
    satisfied <- function(k) {
      below_size <- k < size_safe
      cdf <- binom_cdf(nv_ifelse(below_size, k, 0), size_safe, prob_safe, lower_tail, log_p, op_dtype)
      list(sat = if (lower_tail) cdf >= target else cdf < target, failed = nv_is_nan(cdf))
    }

    # Start from a normal quantile with a skewness correction, as base R does.
    q_safe <- 1 - prob_safe
    sigma <- nv_sqrt(size_safe * prob_safe * q_safe)
    start <- nv_round(cornish_fisher(
      p_safe,
      size_safe * prob_safe,
      sigma,
      (q_safe - prob_safe) / sigma,
      lower_tail,
      log_p
    ))
    found <- discrete_search(start, satisfied, size_safe)

    # Restore invalid and endpoint results. If the iteration cap stopped the
    # search early, the result is the current bound rather than a guarantee
    # that the exact count was found.
    nv_ifelse(
      invalid,
      NaN,
      nv_ifelse(is_zero, 0, nv_ifelse(is_size, size, found))
    )
  },
  static = c("lower_tail", "log_p")
)

## Poisson distribution helpers -------------------------------------------------

# Log Poisson probability for a whole number x >= 0 and finite lambda >= 0.
# For x >= 1, use Loader's saddle point expansion:
#   log p(x) = -stirlerr(x) - bd0(x, lambda) - log(2 pi x) / 2
# At x = 0, return the exact value -lambda instead.
# The expansion is NaN at x = 0, which the select discards. This helper is used
# without automatic differentiation, so the unused branch needs no safe input.
#
# At lambda = 0, the expansion gives -Inf, which is the correct log probability
# for x >= 1. In practice, the callers of `ppois_raw()` (the only caller of this
# helper) replace zero lambda beforehand.
dpois_log_raw <- function(x, lambda) {
  saddle <- stirlerr(x) + bd0(x, lambda)
  nv_ifelse(x == 0, -lambda, -saddle - 0.5 * nv_log(x) - 0.5 * base::log(2 * pi))
}

# Poisson tail series: multiplying the result by p(k) gives P(X > k) for
# upper = TRUE, or P(X < k) otherwise, as R's `pd_upper_series()` and
# `pd_lower_series()`:
#   upper: lambda / (k + 1) + lambda^2 / ((k + 1)(k + 2)) + ...
#   lower: k / lambda + k (k - 1) / lambda^2 + ...  (k terms)
#
# To avoid a division at each step, the sum is carried as num / den and the
# current term as term / den, and divided out once at the end. Power-of-two
# scaling keeps them in range without rounding from the scaling itself.
#
# Inputs must not carry gradients: `nv_while()` has no gradient rule.
poisson_series <- function(k, lambda, upper, is_f32) {
  tol <- if (is_f32) 2^-24 else 2^-53
  # Tuned on the CPU backend for lambda from 1 to 150 (a trip costs about as
  # much as several unrolled steps; frozen steps cost more than plain ones)
  n_unroll <- if (is_f32) 8L else 24L
  # num, den and term are renormalised at least once every `steps_per_trip`
  # steps. Within `max_steps` each scaled a_n is below 2^10, so 8 steps grow
  # them by less than 2^80, which still fits in f32: raising either constant
  # means checking this bound again. Convergence takes at most ~170 terms in
  # f64 and ~90 in f32 in the regions `ppois_raw()` uses the series in.
  steps_per_trip <- 8L
  max_steps <- 1000L
  # u_n = u0 - n du and a_n = a0 + n da, scaled by c so that a0 is in [1, 2)
  # (or a little more where it is beyond the largest power of two). The lower
  # series is finite: its u_n reach zero at n = k, after which the steps only
  # multiply a zero term. With k >= 1 it has lambda > k >= 1, so c <= 1 and
  # n du stays small; but at k = 0 (the only count below a mean under 1) c is
  # about 1 / lambda, and n c overflows for a small enough lambda, making the
  # zero term NaN. The series has no terms there at all, so du is zero.
  a0 <- nv_ifelse(upper, k + 1, lambda)
  c <- inv_pow2(nv_pmin(a0, if (is_f32) 2^126 else 2^1022))
  u0 <- nv_ifelse(upper, lambda, k) * c
  du <- nv_ifelse(upper | (k == 0), 0, c)
  a0 <- a0 * c
  da <- nv_ifelse(upper, c, 0)
  # Put den in [1, 2), scaling num and term by the same exact power of two.
  normalise <- function(f) {
    s <- inv_pow2(f$den)
    list(term = f$term * s, num = f$num * s, den = f$den * s)
  }
  step <- function(f, n, live = NULL) {
    u <- u0 - n * du
    a <- a0 + n * da
    if (!is.null(live)) {
      u <- nv_ifelse(live, u, 0)
      a <- nv_ifelse(live, a, 1)
    }
    term <- f$term * u
    list(term = term, num = f$num * a + term, den = f$den * a)
  }
  # Term ratios are below one and decreasing in the regions `ppois_raw()` uses
  # the series in, so the rest of the series is at most
  # next_term / (1 - next_ratio). Requiring that to be <= sum * tol, with tol
  # half the machine epsilon, gives without dividing
  #   term * u <= (a - u) * num * tol
  # where u and a are those of the next term. Once it holds, term is set to
  # zero, which marks the element converged for the loop below.
  settle <- function(f, n) {
    u <- u0 - (n + 1) * du
    a <- a0 + (n + 1) * da
    f$term <- nv_ifelse(f$term * u <= (a - u) * f$num * tol, 0, f$term)
    f
  }

  # The unrolled steps deliberately do not freeze converged elements: such an
  # element only takes on terms below half an ulp, and freezing doubled the
  # measured cost for no measurable gain in accuracy. The loop does freeze
  # them (u = 0 and a = 1 leave num and den as they are), so that an element's
  # result does not depend on how many trips the others need.
  f <- normalise(list(term = u0, num = u0, den = a0))
  for (n in seq_len(n_unroll)) {
    f <- step(f, n)
    if (n %% steps_per_trip == 0L || n == n_unroll) {
      f <- normalise(f)
    }
  }
  f <- settle(f, n_unroll)
  out <- nv_while(
    init = list(n = nv_scalar_like(k, n_unroll), term = f$term, num = f$num, den = f$den),
    cond = function(n, term, num, den) (n < max_steps) & nv_any(term != 0),
    body = function(n, term, num, den) {
      f <- list(term = term, num = num, den = den)
      for (j in seq_len(steps_per_trip)) {
        f <- settle(step(f, n + j, live = f$term != 0), n + j)
      }
      f <- normalise(f)
      list(n = n + steps_per_trip, term = f$term, num = f$num, den = f$den)
    }
  )
  out$num / out$den
}

# Coefficients of the asymptotic expansion of the Poisson distribution function
# for large k with lambda close to k, as R's `ppois_asymp()`.
ppois_asymp_a <- c(
  2 / 3,
  -4 / 135,
  8 / 2835,
  16 / 8505,
  -8992 / 12629925,
  -334144 / 492567075,
  698752 / 1477701225
)
ppois_asymp_b <- c(
  1 / 12,
  1 / 288,
  -139 / 51840,
  -571 / 2488320,
  163879 / 209018880,
  5246819 / 75246796800,
  -534703531 / 902961561600
)

# P(X <= k) (or P(X > k)) for whole k with lambda close to k, both large, as R's
# `ppois_asymp()`: a normal approximation in the signed root of the deviance,
# corrected by an expansion in 1/k. Returns the probability as `value`. With
# `grad = TRUE` and `log_p = TRUE` it also returns `grad`, the gradient of that
# log probability wrt lambda (ignoring sign): dpois(k) / probability.
ppois_asymp <- function(k, lambda, lower_tail, log_p, is_f32, grad = FALSE) {
  dfm <- lambda - k
  # Near the top of the range the reciprocal of k that XLA divides by would be
  # subnormal, so both are scaled down first (see `bd0()`)
  huge <- k > (if (is_f32) 2^120 else 2^1016)
  t <- nv_ifelse(huge, dfm * 2^-4, dfm) / nv_ifelse(huge, k * 2^-4, k)
  # |t| <= 1/4 in the region this is used in, where 4 terms of the series are
  # accurate to f32 and 9 to f64
  pt <- -log1pmx(t, if (is_f32) 4L else 9L)
  # As two square roots, as 2 * k can overflow
  s2pt <- nv_sqrt(2 * pt) * nv_sqrt(k)
  s2pt <- nv_ifelse(dfm < 0, -s2pt, s2pt)

  res12 <- 0
  res1_term <- res1_ig <- nv_sqrt(k)
  res2_term <- res2_ig <- s2pt
  for (i in seq_along(ppois_asymp_a)) {
    res12 <- res12 + res1_ig * ppois_asymp_a[[i]] + res2_ig * ppois_asymp_b[[i]]
    res1_term <- res1_term * pt / i
    res2_term <- res2_term * 2 * pt / (2 * i + 1)
    res1_ig <- res1_ig / k + res1_term
    res2_ig <- res2_ig / k + res2_term
  }
  elfb <- k + horner(1 / k, rev(ppois_asymp_b))
  if (!lower_tail) {
    elfb <- -elfb
  }
  f <- res12 / elfb

  np <- nv_pnorm(s2pt, lower_tail = !lower_tail, log_p = log_p)
  log_dnorm <- -0.5 * s2pt * s2pt - 0.5 * base::log(2 * pi)
  if (!log_p) {
    return(list(value = np + f * nv_exp(log_dnorm)))
  }
  # `nv_pnorm(log_p = TRUE)` takes log1p(-erfc) above one half, which XLA's
  # CPU f64 `log1p()` gets wrong by up to ~120 ulp for probabilities in about
  # (0.586, 0.64). Fixed upstream in openxla/xla#46765, so deliberately not
  # worked around.
  value <- np + nv_log1p(f * nv_exp(log_dnorm - np))
  if (!grad) {
    return(list(value = value))
  }

  # The gradient, dpois(k) / tail, for the small tail only (z > 0); the caller
  # does the other. Both are tiny here, ~exp(-z^2 / 2), and subtracting their
  # logs would lose ~z^2 ulps, so write both via the normal curve and cancel
  # the tiny parts on paper. With m = phi(z) / Phi(-z):
  #   tail = Phi(-z) (1 + f m)
  #   dpois(k) = phi(z) exp(-stirlerr(k)) / sqrt(k)  (s^2 / 2 is bd0(k, lambda))
  #   dpois(k) / tail = m exp(-stirlerr(k)) / (sqrt(k) (1 + f m))
  # where m = sqrt(2 / pi) / erfcx(z / sqrt(2)) is a moderate number.
  z <- if (lower_tail) s2pt else -s2pt
  z_small <- nv_ifelse(z > 0, z, 1) * base::sqrt(0.5)
  m <- base::sqrt(2 / pi) / erfcx(z_small, if (is_f32) 8 else 20, if (is_f32) 6L else 8L)
  # stirlerr(k) minus its lookup table: k is always large here, and the lookup
  # would stop XLA vectorising this code on CPU
  coefs <- if (is_f32) stirlerr_series[4:7] else stirlerr_series
  k_series <- nv_pmax(k, 16)
  stirlerr_k <- horner(1 / (k_series * k_series), coefs) / k_series
  grad <- m * nv_exp(-stirlerr_k) / (nv_sqrt(k) * (1 + f * m))
  list(value = value, grad = grad)
}

# The Poisson distribution function at a whole k >= 0 and finite lambda > 0,
# with the regions of R's `pgamma_raw()` (for its shape k + 1 and x = lambda):
#   - lambda well below k: P(X > k) = p(k) * (upper series);
#   - lambda well above k: P(X <= k) = p(k) * (1 + lower series);
#   - otherwise k and lambda are close and large: the asymptotic expansion.
# Each series gives one tail directly and the other as its complement, which is
# taken on the log scale to keep full accuracy either side. Returns the
# probability, log dpois(k), and which elements took a series (whose error
# `nv_qpois()` allows for). With `grad = TRUE` it also returns the gradient's
# size wrt lambda: dpois(k), or on the log scale dpois(k) / probability. It
# comes as two factors, `grad` and `grad_factor`, so each stays finite when
# their product overflows.
# Due to while loop in series, no gradient support.
ppois_raw <- function(k, lambda, lower_tail, log_p, is_f32, grad = FALSE) {
  upper <- (lambda <= k) & (lambda < 0.8 * (k + 51))
  lower <- (k < lambda) & (k + 1 < 0.8 * (lambda + 50))
  use_series <- upper | lower

  log_density <- dpois_log_raw(k, lambda)

  # The asymptotic expansion is computed only if some element needs it (see
  # `skip_unless()`). The series is not gated the same way, though it costs
  # about a third of the time of a batch that is all asymptotic (large lambda
  # with counts near it): its result then has to cross a loop boundary, which
  # on the CPU backend cost up to ~18% (f64) on batches that need only the
  # series (lambda below ~200, the common case for count data) at 1e7
  # elements, ~3-5% at 1e6.
  series_branch <- function() {
    # Elements outside both series are given k = 0, whose lower series has no
    # terms for any finite lambda > 0 (see `poisson_series()`), so lambda is
    # passed as it is and a scalar lambda stays scalar
    sum <- poisson_series(nv_ifelse(use_series, k, 0), lambda, upper, is_f32)
    # The upper series starts at lambda / (k + 1), which underflows for a tiny
    # enough lambda (below the smallest normal times k + 1), leaving the sum zero
    # and its log -Inf. The later terms are then below eps of it, so the series
    # is lambda / (k + 1) to working precision, and its log is taken directly.
    # Only computed if some element needs it.
    smallest_normal <- if (is_f32) 2^-126 else 2^-1022
    underflowed <- upper & (sum < smallest_normal)
    log_upper_sum <- nv_log(nv_ifelse(upper, sum, 1))
    log_upper_sum <- skip_unless(nv_any(underflowed), log_upper_sum, function() {
      nv_ifelse(
        underflowed,
        -log_ratio(nv_ifelse(underflowed, k + 1, 1), nv_ifelse(underflowed, lambda, 1)),
        log_upper_sum
      )
    })
    log_tail <- log_density + nv_ifelse(upper, log_upper_sum, nv_log1p(sum))
    # `log_tail` is the log of P(X > k) for the upper series, P(X <= k) for the lower
    direct <- if (lower_tail) lower else upper
    from_series <- if (log_p) {
      nv_ifelse(direct, log_tail, log1mexp(log_tail))
    } else {
      nv_ifelse(direct, nv_exp(log_tail), -nv_expm1(log_tail))
    }
    out <- list(value = from_series)
    if (grad && log_p) {
      # The gradient, dpois(k) / tail, without logs. The tail the series sums
      # is dpois(k) times the sum (plus one for the lower), so the gradient is
      # just 1 / sum; subtracting logs would lose |log tail| ulps. If the upper
      # sum underflowed it equals lambda / (k + 1), whose reciprocal can
      # overflow, so it's kept as 1 / lambda times k + 1. The other tail is at
      # least about a half, where subtracting logs is fine.
      out$grad <- nv_ifelse(
        direct,
        1 / nv_ifelse(underflowed, lambda, nv_ifelse(upper, sum, 1 + sum)),
        nv_exp(log_density - from_series)
      )
      out$grad_factor <- nv_ifelse(direct & underflowed, k + 1, 1)
    }
    out
  }
  series <- series_branch()
  # The series' elements are given k = lambda. Where lambda is tiny that can
  # overflow inside the expansion, but only on elements that are then not
  # selected, and the expansion is taken without gradient.
  asymp_grad <- grad && log_p
  asymp_init <- list(value = nv_fill_like(k, 0))
  if (asymp_grad) {
    asymp_init$grad <- nv_fill_like(k, 0)
  }
  asymp <- skip_unless(nv_any(!use_series), asymp_init, function() {
    ppois_asymp(
      nv_ifelse(use_series, lambda, k),
      lambda,
      lower_tail,
      log_p,
      is_f32,
      grad = asymp_grad
    )
  })
  out <- list(
    value = nv_ifelse(use_series, series$value, asymp$value),
    log_density = log_density,
    use_series = use_series
  )
  if (grad) {
    if (log_p) {
      # The expansion's gradient is for the small tail only. The other tail is
      # at least about a half, where subtracting logs is fine, and a little more
      # accurate (log dpois(k) comes from bd0(), which beats the expansion's)
      small_tail <- if (lower_tail) lambda > k else lambda < k
      out$grad <- nv_ifelse(
        use_series,
        series$grad,
        nv_ifelse(small_tail, asymp$grad, nv_exp(log_density - out$value))
      )
      out$grad_factor <- nv_ifelse(use_series, series$grad_factor, 1)
    } else {
      out$grad <- nv_exp(log_density)
      out$grad_factor <- nv_scalar_like(lambda, 1)
    }
  }
  out
}

#' @title The Poisson Distribution
#' @name nv_poisson
#' @description
#' Density (`nv_dpois`), distribution function (`nv_ppois`), and quantile
#' function (`nv_qpois`) for the Poisson distribution with mean `lambda`.
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`). These are counts, given as integers or floats; see
#'   Non-Integer Values for values that are not whole numbers.
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`. With `log_p = TRUE`, supply log-probabilities
#'   in \eqn{[-\infty, 0]} instead.
#' @param lambda ([`arrayish`])\cr
#'   Mean of the distribution. Scalars broadcast; otherwise `lambda` and
#'   `x`/`q`/`p` must have the same shape.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the densities/probabilities are given as logarithms. For
#'   `nv_qpois` this describes the input `p`.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le x)}; otherwise,
#'   \eqn{P(X > x)}.
#' @details
#' The Poisson distribution has probability mass function:
#' \deqn{p(x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0, 1, 2, \ldots}
#' where \eqn{\lambda} is `lambda`.
#'
#' Invalid arguments give `NaN`, as in base R, but without a warning (that is
#' a negative `lambda`). Zero and infinite `lambda` follow base R's endpoint
#' conventions:
#' * `lambda = 0`: the point mass at zero.
#' * `lambda = Inf`: densities are zero. The lower-tail probability is zero at
#'   finite `q` and one at `q = Inf`. Quantiles are `NaN`.
#'
#' `nv_dpois` uses the saddle point expansion of
#' `r cite_bib("loader2000fast")`, as base R does. `nv_ppois` uses the
#' incomplete gamma function methods of base R's [stats::pgamma()]: a series in
#' the tails, and an asymptotic expansion where `q` and `lambda` are large and
#' close. `nv_qpois` finds the smallest \eqn{x}
#' with \eqn{P(X \le x) \ge p}, starting from a Cornish-Fisher approximation as
#' base R does. Above `2^24` in `f32` and `2^53` in `f64` not every whole number
#' is a float, and `nv_qpois` then returns the smallest float \eqn{x} with
#' \eqn{P(X \le x) \ge p}.
#'
#' @templateVar dist pois
#' @templateVar params `lambda`
#' @template section_distribution_nonint
#' @section Gradients:
#' `nv_dpois` and `nv_ppois` can be differentiated with respect to `lambda`.
#' For `nv_ppois` only the first derivative is correct: second and higher
#' derivatives are returned as zero. At `lambda = 0`, where
#' only a right derivative exists, the implementation returns it for
#' `nv_dpois(0, lambda)` and, by convention, zero for the other probabilities
#' at `lambda = 0`. `nv_qpois` is a step function, and autodiff returns a zero
#' gradient for all of its arguments.
#'
#' @template section_distribution_dtype
#' @references
#' `r format_bib("loader2000fast")`
#' @return ([`arrayish`])\cr
#' `nv_dpois()`, `nv_ppois()`, and `nv_qpois()` return an [`arrayish`] with the
#' float data type described under Data Types and the shape after scalar
#' broadcasting with `lambda`.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 1, 2, 5, 10))
#' nv_dpois(x, lambda = 4)
#' nv_dpois(x, lambda = 4, log = TRUE)
#'
#' nv_ppois(x, lambda = 4)
#' nv_ppois(x, lambda = 4, lower_tail = FALSE)
#' nv_ppois(x, lambda = 4, log_p = TRUE)
#'
#' p <- nv_array(c(0.025, 0.5, 0.975))
#' nv_qpois(p, lambda = 4)
#' nv_qpois(p, lambda = 4, lower_tail = FALSE)
#' nv_qpois(
#'   nv_array(c(-700, -2, -0.1), dtype = "f64"),
#'   lambda = 1000,
#'   log_p = TRUE
#' )
NULL

#' @rdname nv_poisson
#' @export
nv_dpois <- jit(
  function(x, lambda, log = FALSE) {
    assert_flag(log)
    args <- promote_distribution_args(x = x, lambda = lambda)
    x <- args$x
    lambda <- args$lambda
    assert_rng_float_dtype(dtype(x), arg = "x")

    # As base R, `x` is rounded to the nearest whole number, once anything not
    # within a relative 1e-9 of one, and infinite `x` (which `is_nonint()`
    # counts as whole, as base R does), has been given probability zero. Only
    # the rounded value is used, so the gradient wrt `x` is zero; it is detached
    # so that the NaN of an infinite `x` cannot reach it through the test.
    x <- stop_gradient(x)
    x_whole <- nv_round(x)
    # The sign is tested before rounding, as base R does: a tiny negative `x`
    # rounds to zero but is outside the support
    zero <- is_nonint(x) | !nv_is_finite(x) | (x < 0) | (lambda == Inf) | ((lambda == 0) & (x_whole != 0))
    valid <- !nv_is_nan(x) & (lambda >= 0)

    # Each branch is fed a safe stand-in where it is not selected, so it cannot
    # poison the gradient. lambda is guarded on its own conditions alone, so a
    # scalar lambda stays scalar: where lambda is valid the branches are finite
    # at the stand-in x anyway.
    at_zero <- x_whole == 0
    pos <- valid & !zero & !at_zero
    x_pos <- nv_ifelse(pos, x_whole, 1)
    lambda_pos <- nv_ifelse((lambda > 0) & (lambda != Inf), lambda, 1)
    lambda_zero <- nv_ifelse((lambda >= 0) & (lambda != Inf), lambda, 0)

    saddle <- stirlerr(x_pos) + bd0(x_pos, lambda_pos)
    density <- if (log) {
      nv_ifelse(at_zero, -lambda_zero, -saddle - 0.5 * nv_log(x_pos) - 0.5 * base::log(2 * pi))
    } else {
      # sqrt(2 pi x) as two factors, which cannot overflow
      nv_ifelse(at_zero, nv_exp(-lambda_zero), nv_exp(-saddle) / (base::sqrt(2 * pi) * nv_sqrt(x_pos)))
    }
    nv_ifelse(valid, nv_ifelse(zero, if (log) -Inf else 0, density), NaN)
  },
  static = "log"
)

#' @rdname nv_poisson
#' @export
nv_ppois <- jit(
  function(q, lambda, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(q = q, lambda = lambda)
    q <- args$q
    lambda <- args$lambda
    op_dtype <- assert_rng_float_dtype(dtype(q), arg = "q")
    is_f32 <- op_dtype == "f32"

    # Resolved directly, in base R's order: P(X <= q) is 0 below the support,
    # 1 for `lambda = 0` or `q = Inf`, and 0 for `lambda = Inf`.
    below <- q < 0
    at_one <- (lambda == 0) | (q == Inf)
    valid <- !nv_is_nan(q) & (lambda >= 0)
    regular <- valid & !below & !at_one & (lambda != Inf)
    dt0 <- dt_zero(lower_tail, log_p)
    dt1 <- dt_one(lower_tail, log_p)

    # Values just below a whole number count as that number, as in base R
    k <- whole_count(stop_gradient(nv_ifelse(regular, q, 0)))
    # Guarded on lambda's own conditions, so a scalar lambda stays scalar; the
    # elements it replaces are among those resolved directly below
    lambda_reg <- nv_ifelse((lambda > 0) & (lambda != Inf), lambda, 1)
    raw <- ppois_raw(k, stop_gradient(lambda_reg), lower_tail, log_p, is_f32, grad = TRUE)

    # The summation is not differentiable, so the exact derivative wrt `lambda`,
    # -/+ dpois(k) (divided by the probability on the log scale), is attached
    # by adding zero times it. Only the first derivative is right this way (see
    # `stop_gradient()`): no differentiable function has this derivative but
    # the distribution function itself.
    # The derivative can overflow on the log scale (it's about (k + 1) / lambda
    # for a tiny lambda), and 0 * Inf would make the value NaN. So the zero is
    # multiplied by its two finite factors one at a time: the value stays exact,
    # and only the gradient, their product, overflows. A non-finite factor
    # gives a zero gradient.
    finite <- nv_is_finite(raw$grad) & nv_is_finite(raw$grad_factor)
    grad <- nv_ifelse(finite, raw$grad, 0)
    delta <- lambda_reg - stop_gradient(lambda_reg)
    delta <- if (lower_tail) -delta else delta
    value <- raw$value + (delta * nv_ifelse(finite, raw$grad_factor, 0)) * grad

    res <- nv_ifelse(below, dt0, nv_ifelse(at_one, dt1, nv_ifelse(lambda == Inf, dt0, value)))
    nv_ifelse(valid, res, NaN)
  },
  static = c("lower_tail", "log_p")
)

#' @rdname nv_poisson
#' @export
nv_qpois <- jit(
  function(p, lambda, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(p = p, lambda = lambda)
    op_dtype <- assert_rng_float_dtype(dtype(args$p), arg = "p")
    # A step function has a zero gradient, and the search below is a while
    # loop, which has none at all
    p <- stop_gradient(args$p)
    lambda <- stop_gradient(args$lambda)
    is_f32 <- op_dtype == "f32"
    eps <- if (is_f32) 2^-23 else 2^-52
    max_float <- if (is_f32) (2 - 2^-23) * 2^127 else .Machine$double.xmax

    # Resolved directly, in base R's order: NaN outside the admissible range,
    # 0 for `lambda = 0` and at the bottom of the probability scale, and Inf
    # at the top
    in_range <- if (log_p) p <= 0 else (p >= 0) & (p <= 1)
    at_bottom <- p == dt_zero(lower_tail, log_p)
    at_top <- p == dt_one(lower_tail, log_p)
    valid <- in_range & nv_is_finite(lambda) & (lambda >= 0)
    zero_result <- (lambda == 0) | at_bottom
    regular <- valid & !zero_result & !at_top
    p <- nv_ifelse(regular, p, if (log_p) -1 else 0.5)
    # Guarded on lambda's own conditions, so a scalar lambda stays scalar; the
    # elements it replaces are among those resolved directly above
    lambda <- nv_ifelse((lambda > 0) & nv_is_finite(lambda), lambda, 1)

    # Cornish-Fisher starting value. base R rounds it, but it lands about a half
    # above the quantile of a `p` at a jump, so rounding often starts the
    # search one too high; rounded down, the quantile is nearly always the start
    # or the next count, which the search settles in two evaluations. The start
    # does not change the answer.
    sigma <- nv_sqrt(lambda)
    start <- nv_floor(cornish_fisher(p, lambda, sigma, 1 / sigma, lower_tail, log_p))

    target <- quantile_target(p, lower_tail, log_p)
    # In f64 the error of the distribution function itself also needs allowing
    # for, so that a `p` from base R at a jump is not missed: relative to the
    # smaller of the two tails it grows like (1 + |log tail|) ulp, plus
    # |y - lambda| ulp where a series was summed (from bd0()'s x log(x / np)).
    # The factor 16 is an empirical allowance, not a proven bound: measured
    # against base R for lambda from 0.5 to 1e14, and against MPFR at the jumps
    # for lambda up to 1e8, the error stayed under 9 such ulp in the series and
    # under 4 in the asymptotic expansion. Any wider and a `p` just above a jump
    # could be read as on it, giving a quantile one too low. In f32, rounding
    # `p` to f32 displaces it further than this, and a wider fuzz only costs
    # agreement elsewhere.
    if (!is_f32) {
      log_upper <- if (log_p) log1mexp(p) else nv_log1p(-p)
      log_lower <- if (log_p) p else nv_log(p)
      log_small_tail <- nv_pmin(log_lower, log_upper)
      # How far `p` may move per unit of relative error in the smaller tail
      scale <- if (log_p) nv_pmin(-p, 1) else nv_pmin(p, 1 - p)
    }
    # Whether the quantile is at or below y. The largest float is taken as
    # satisfying (by `discrete_search()`). Below lambda = the largest float that
    # changes nothing: the condition holds a few sd above lambda, far less than
    # an ulp there, so the largest float already satisfies it wherever the
    # search reaches it. At lambda = the largest float there may be no finite
    # float that satisfies it, and this returns the largest float instead of
    # Inf: a saturating convention, as base R's.
    satisfied <- function(y) {
      raw <- ppois_raw(y, lambda, lower_tail, log_p, is_f32)
      threshold <- target
      if (!is_f32) {
        ulps <- 1 - log_small_tail + nv_ifelse(raw$use_series, nv_abs(y - lambda), 0)
        extra <- nv_pmin(16 * eps * ulps, 0.25) * scale
        adjusted <- if (lower_tail) threshold - extra else threshold + extra
        # As in `quantile_target()`, the allowance is not applied where it
        # would overflow, or flush a tiny `p` to zero
        threshold <- nv_ifelse(if (log_p) nv_is_finite(adjusted) else adjusted > 0, adjusted, threshold)
      }
      list(sat = if (lower_tail) raw$value >= threshold else raw$value < threshold)
    }
    found <- discrete_search(start, satisfied, max_float)

    res <- nv_ifelse(zero_result, 0, nv_ifelse(at_top, Inf, found))
    nv_ifelse(valid, res, NaN)
  },
  static = c("lower_tail", "log_p")
)
