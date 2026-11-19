## ---------------------------------------------------------------------------
## nv_qpois against base R qpois().
##
## nv_qpois() starts from base R's Cornish-Fisher approximation, rounded down,
## and searches for the smallest count whose distribution function reaches p,
## stepping out by doubling and then bisecting, each step an evaluation of
## nv_ppois()'s ppois_raw(). Before the search p is moved by base R's
## tolerance, 8 eps (2 eps on the log scale) at the cell's precision, and in
## f64 by a further allowance for the distribution function's own error, so
## that a p computed at a support point maps back to it.
##
## The quantile is a step function, so a disagreement is a whole count, and it
## concentrates where p is within that tolerance of a step: there anvl and base
## R evaluate the distribution function each in their own way, and the one
## that lands on the other side of the tolerance picks the neighbouring count.
## The steps at the median count and five standard deviations below it are
## checked exactly, with their neighbours.
##
## base R's qpois() fails on the log scale below log p ~ -9e307 (~-9e292 at the
## `tiny` lambda): its Cornish-Fisher start squares a z that overflows, and it
## returns Inf where the quantile is 0 (lower tail) or a finite ~1e305 (upper
## tail). nv_qpois() clamps z and returns them. No stable reference recognises
## this, so those samples are counted against anvl: a discrete quantile's
## reference would need an MPFR truth that reproduces base R's tolerance at
## every step, where a count either way is ~1e12 ulps. The lower tail is easily
## checked by hand: log P(X <= 0) = -lambda, so for every lambda in these sets
## the quantile of any finite log p below -1e6 is 0.
##
## There are no gradient cells: the quantile is piecewise constant, and
## nv_qpois() detaches its inputs, so its gradient is zero by construction. JAX
## has no Poisson quantile, so there is no twin.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_poisson.R"), local = TRUE)

sweep_spec(
  name = "nv_qpois",
  family = "poisson",
  params = POIS_PQ_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),

  ## A quantile function is only defined on its probability scale; base R
  ## returns NaN with a warning off it. A value disagreement there is still a
  ## failure (NaN specified, something else returned).
  domain = function(p, f) if (isTRUE(f$log_p)) c(-Inf, 0) else c(0, 1),
  ## Not algorithm switches but the function's own steps, where the search's
  ## tolerance decides: the distribution function at the median count and at
  ## five standard deviations below it, by base R. See R/api-distributions.R,
  ## nv_qpois.
  branch_points = function(p, f, dtype) {
    l <- p$lambda
    k <- unique(pmax(0, floor(c(median = l, tail = l - 5 * sqrt(l)))))
    setNames(
      ppois(k, l, lower.tail = f$lower_tail, log.p = f$log_p),
      paste0("cdf_step_", k)
    )
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_qpois(
      anvl::nv_array(x, dtype = dtype),
      p$lambda,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    suppressWarnings(qpois(x, p$lambda, lower.tail = f$lower_tail, log.p = f$log_p))
  }
)
