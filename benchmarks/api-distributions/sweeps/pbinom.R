## ---------------------------------------------------------------------------
## nv_pbinom against base R pbinom().
##
## nv_pbinom() is the incomplete beta function, P(X > k) = I_p(k + 1, n - k),
## by the methods of TOMS 708 that base R's pbeta() chooses between: the
## continued fraction bfrac() (here division-free and unrolled), and the
## asymptotic expansion basym() for large a and b near the mean. The leading
## factor is the saddle point density. Over q, each parameter set crosses the
## switch between the two tails and, where it has one, both seams of the
## expansion (see _binomial.R).
##
## A stable reference covers the lower-tail log cells. There base R's pbeta()
## underflows in the far lower tail and returns -Inf for an ordinary log
## probability (-861 at q = 32 for size 1e6, prob 1e-3), which would otherwise
## be counted against anvl; see binom_log_cdf_lower() in _binomial.R. In the
## other variants and away from that tail base R is accurate to a few ulp, and
## the disagreements a sweep finds are the shared conditioning of a probability
## formed as exp of a log. At sizes beyond these sets base R also returns
## impossible positive log probabilities; the unit tests check those against
## exact MPFR sums.
##
## There are no gradient cells: nv_pbinom() iterates in an nv_while() loop,
## which gradient() cannot differentiate.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_binomial.R"), local = TRUE)

sweep_spec(
  name = "nv_pbinom",
  family = "binomial",
  params = BINOM_PQ_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),

  ## pbinom() is defined on the whole line, so nothing below or above the
  ## support is excused: a disagreement there is a real finding.
  domain = function(p, f) c(-Inf, Inf),
  support = function(p, f) c(0, p$size),
  ## The switches between the two tails and into and out of the asymptotic
  ## expansion (binom_cdf_switches(), _binomial.R), and base R's fuzz:
  ## floor(q + 1e-7) takes q = 1 - 1e-7 to 1. See R/api-distributions.R,
  ## binom_cdf and nv_pbinom.
  branch_points = function(p, f, dtype) {
    c(binom_cdf_switches(p$size, p$prob), floor_fuzz = 1 - 1e-7)
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_pbinom(
      anvl::nv_array(x, dtype = dtype),
      p$size,
      p$prob,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    pbinom(x, p$size, p$prob, lower.tail = f$lower_tail, log.p = f$log_p)
  },
  ref_stable = function(x, p, f) binom_log_cdf_lower(x, p$size, p$prob),
  ref_stable_mpfr = function(x, p, f) mp_log_pbinom_lower(x, p$size, p$prob),
  ## Where it is base R's value it carries base R's error, measured against
  ## MPFR at up to ~2000 ulps of the log far above the mode: the log there is a
  ## tiny -P(X > k), around 1e-246, and any double evaluation of a tail that
  ## small through exp of its log carries ~|log| ulps. The far series: the
  ## stable log density, a few ulps of a log of the size of the result, and the
  ## log of a sum in [1, 10]: ~8. 4096 covers the former, as for the density;
  ## it is checked, not assumed. base R's failures in the far tail are -Inf or
  ## a log off by hundreds, which no bound covers.
  ref_stable_bound_ulp64 = 4096,
  ref_stable_covers = function(f) isTRUE(f$lower_tail) && isTRUE(f$log_p),
  ref_stable_note = "base R's pbeta() fails in the far lower tail: -Inf for an ordinary log probability (log P = -3.6e5 at q = 25 for size 1e6, prob 0.3), or a finite but wrong one (-603 for -844 at q = 37 for size 1e6, prob 1e-3)",

  ## The betainc twin (_binomial.R) covers all four variants, the log scale as
  ## the log of the probability, which is what that route offers.
  jax_value = function(x, dtype, p, f) jax_pbinom_value(x, dtype, p, f)
)
