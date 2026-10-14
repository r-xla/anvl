## ---------------------------------------------------------------------------
## nv_qbinom against base R qbinom().
##
## nv_qbinom() starts from base R's Cornish-Fisher approximation and searches
## for the smallest count whose distribution function reaches p, stepping out
## by doubling and then bisecting, each step an evaluation of nv_pbinom()'s
## binom_cdf(). Before the search p is moved by base R's tolerance, 8 eps (2 eps
## on the log scale) at the cell's precision, so that a p computed at a
## support point maps back to it.
##
## The quantile is a step function, so a disagreement is a whole count, and it
## concentrates where p is within that tolerance of a step: there anvl and base
## R evaluate the distribution function each in their own way, and the one
## that lands on the other side of the tolerance picks the neighbouring count.
## The steps at the median count and five standard deviations below it are
## checked exactly, with their neighbours.
##
## There are no gradient cells: the quantile is piecewise constant, and the
## search is an nv_while() loop, which gradient() cannot differentiate. JAX
## has no binomial quantile, so there is no twin.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_binomial.R"), local = TRUE)

sweep_spec(
  name = "nv_qbinom",
  family = "binomial",
  params = BINOM_PQ_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),

  ## A quantile function is only defined on its probability scale; base R
  ## returns NaN with a warning off it. A value disagreement there is still a
  ## failure (NaN specified, something else returned).
  domain = function(p, f) if (isTRUE(f$log_p)) c(-Inf, 0) else c(0, 1),
  ## Not algorithm switches but the function's own steps, where the search's
  ## tolerance decides: the distribution function at the median count and at
  ## five standard deviations below it, by base R. See R/api-distributions.R,
  ## nv_qbinom.
  branch_points = function(p, f, dtype) {
    n <- p$size
    pr <- p$prob
    k <- unique(pmax(0, floor(c(median = n * pr, tail = n * pr - 5 * sqrt(n * pr * (1 - pr))))))
    setNames(
      pbinom(k, n, pr, lower.tail = f$lower_tail, log.p = f$log_p),
      paste0("cdf_step_", k)
    )
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_qbinom(
      anvl::nv_array(x, dtype = dtype),
      p$size,
      p$prob,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    suppressWarnings(qbinom(x, p$size, p$prob, lower.tail = f$lower_tail, log.p = f$log_p))
  }
)
