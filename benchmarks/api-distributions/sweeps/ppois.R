## ---------------------------------------------------------------------------
## nv_ppois against base R ppois().
##
## nv_ppois() follows the regions of base R's pgamma_raw() for shape k + 1 at
## lambda: the upper series where lambda is well below k, the lower series
## where it is well above, and the asymptotic expansion ppois_asymp() where
## both are large and close. Each series gives one tail directly and the other
## by its complement on the log scale. Over q, each parameter set crosses the
## switches between them (see _poisson.R).
##
## A stable reference covers the upper-tail log cells. There base R's ppois()
## returns -Inf for q from ~2.5e305 at the `small` lambda, where log P(X > q)
## is a finite ~-1.7975e308, which would otherwise be counted against anvl; see
## pois_log_upper_stable() in _poisson.R. In the other variants and away from
## that band base R is accurate to a few ulps in the bulk, and the
## disagreements a sweep finds in the tails are the shared conditioning of a
## probability formed as exp of a log.
##
## The summation is not differentiable, so nv_ppois() attaches the exact first
## derivative in lambda, -/+ p(k) (over the probability on the log scale), and
## the gradient cells check that. q has no derivative: the distribution
## function is a step function in it.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_poisson.R"), local = TRUE)

## With k = floor(q + 1e-7), and p(k) the density, d/dlambda p(j) =
## p(j - 1) - p(j), which telescopes:
##
##   P(X <= k):  -p(k)               log:  -p(k) / P(X <= k)
##   P(X > k):    p(k)               log:   p(k) / P(X > k)
##
## p(k) is base R's, accurate to a few ulps at these sets' round lambdas, and
## on the log scale the quotient is taken by pois_density_over_tail()
## (_poisson.R), from the tail series wherever they converge. Below the support and at q = Inf the value is a
## constant, which both nv_ppois() and this reference resolve directly, with a
## zero derivative.
ppois_grad_ref <- function(q, p, f) {
  lam <- p$lambda
  lower <- isTRUE(f$lower_tail)
  s <- if (lower) -1 else 1
  k <- floor(q + 1e-7)
  out <- rep(0, length(q))
  out[is.na(q)] <- NaN
  i <- which(!is.na(q) & q >= 0 & q < Inf)
  if (length(i)) {
    out[i] <- s *
      if (isTRUE(f$log_p)) {
        pois_density_over_tail(k[i], lam, lower)
      } else {
        dpois(k[i], lam)
      }
  }
  list(lambda = out)
}

grad_ppois <- anvl::jit(
  anvl::gradient(
    \(q, lambda, lower_tail = TRUE, log_p = FALSE) {
      sum(anvl::nv_ppois(q, lambda, lower_tail = lower_tail, log_p = log_p))
    },
    wrt = "lambda"
  ),
  static = c("lower_tail", "log_p")
)

sweep_spec(
  name = "nv_ppois",
  family = "poisson",
  params = POIS_PQ_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),

  ## ppois() is defined on the whole line, so nothing below or above the
  ## support is excused: a disagreement there is a real finding.
  domain = function(p, f) c(-Inf, Inf),
  support = function(p, f) c(0, Inf),
  ## The switches between the series and the asymptotic expansion
  ## (pois_cdf_switches(), _poisson.R); where the upper series' first term
  ## lambda / (k + 1) falls below the smallest normal and its log is taken
  ## directly, which only the `tiny` lambda reaches; and base R's fuzz:
  ## floor(q + 1e-7) takes q = 1 - 1e-7 to 1. See R/api-distributions.R,
  ## ppois_raw and whole_count.
  branch_points = function(p, f, dtype) {
    under <- p$lambda / (if (dtype == "f32") 2^-126 else 2^-1022)
    top <- if (dtype == "f32") (2 - 2^-23) * 2^127 else .Machine$double.xmax
    c(
      pois_cdf_switches(p$lambda),
      if (under <= top) c(series_underflow = under),
      floor_fuzz = 1 - 1e-7
    )
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_ppois(
      anvl::nv_array(x, dtype = dtype),
      p$lambda,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    ppois(x, p$lambda, lower.tail = f$lower_tail, log.p = f$log_p)
  },
  ref_stable = function(x, p, f) pois_log_upper_stable(x, p$lambda),
  ref_stable_mpfr = function(x, p, f) {
    qn <- mp_num(x)
    out <- mp_fill(x, NaN)
    out[!is.nan(qn) & qn < 0] <- 0
    out[!is.nan(qn) & qn == Inf] <- -Inf
    i <- which(!is.nan(qn) & qn >= 0 & qn < Inf)
    if (length(i)) {
      out[i] <- mp_pois_parts(floor(qn[i] + 1e-7), p$lambda)$log_upper
    }
    out
  },
  ## Where it is base R's value it carries base R's error: up to ~800 ulps of
  ## the log far below the mode, where the log is a tiny -P(X <= k) formed
  ## through exp of its log, as for nv_pbinom(). The far series: the stable log
  ## density, a few ulps of a log of the size of the result, two logs of size
  ## ~700 and the log of a sum in [1, 10]: ~4. 4096 covers the former, as for
  ## the density; it is checked, not assumed. base R's failure in the band is
  ## -Inf for a finite log, which no bound covers.
  ref_stable_bound_ulp64 = 4096,
  ref_stable_covers = function(f) !isTRUE(f$lower_tail) && isTRUE(f$log_p),
  ref_stable_note = "base R's ppois() returns -Inf for q from ~2.5e305 at lambda = 4, where log P(X > q) is a finite ~-1.7975e308: its deviance term overflows before the whole does",

  grad_wrt = "lambda",
  grad = function(x, dtype, p, f) {
    d <- grad_ppois(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$lambda, length(x)), dtype = dtype),
      lower_tail = f$lower_tail,
      log_p = f$log_p
    )
    list(lambda = as.double(d$lambda))
  },
  ref_grad = ppois_grad_ref,
  ## The tail series, whose roundings mostly cancel but are bounded by
  ## 2 r / (1 - r) ulps, ~200 at its cut r = 0.99; or, near the mean, the
  ## quotient of base R's values, a few ulps each (pois_density_over_tail()).
  ## The probability-scale form is base R's density. 256 leaves margin.
  ref_grad_bound_ulp64 = 256,
  ref_grad_mpfr = function(x, p, f) {
    qn <- mp_num(x)
    lower <- isTRUE(f$lower_tail)
    out <- mp_fill(x, 0)
    out[is.nan(qn)] <- NaN
    i <- which(!is.nan(qn) & qn >= 0 & qn < Inf)
    if (length(i)) {
      m <- mp_pois_parts(floor(qn[i] + 1e-7), p$lambda)
      ## p(k) over the tail, from the summed ratio where it is that tail's and
      ## from the logs elsewhere, where log p(k) is of moderate size
      over <- if (!isTRUE(f$log_p)) {
        exp(m$log_density)
      } else if (lower) {
        v <- exp(m$log_density - m$log_lower)
        v[!m$series] <- 1 / m$ratio[!m$series]
        v
      } else {
        v <- exp(m$log_density - m$log_upper)
        v[m$series] <- 1 / m$ratio[m$series]
        v
      }
      out[i] <- if (lower) -over else over
    }
    list(lambda = out)
  },

  ## The incomplete gamma twin (_poisson.R) covers all four variants, the log
  ## scale as the log of the probability, which is what that route offers.
  jax_value = function(x, dtype, p, f) jax_ppois_value(x, dtype, p, f),
  jax_grad = function(x, dtype, p, f) jax_ppois_grad(x, dtype, p, f)
)
