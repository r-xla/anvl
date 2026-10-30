## ---------------------------------------------------------------------------
## Shared by the three Poisson specs. Not a spec itself (leading underscore).
##
## The density is swept in `lambda`, not in x, for the binomial density's
## reason (_binomial.R): a bit-pattern sweep of x lands on a whole number only
## from 2^23 (f32) or 2^52 (f64) upwards, so it would test almost nothing but
## the zero density at non-whole x. Swept in lambda, with x fixed, the density
## is a smooth function on [0, Inf), its gradient with respect to lambda is
## the one anvl differentiates, and lambda sweeps past x, crossing the seams of
## the deviance term bd0(x, lambda). The distribution and quantile functions
## are swept in q and p as usual.
##
## Density parameter sets, x:
##
##   at_zero    0. The exact log density -lambda; no saddle point expansion.
##   one        1. At lambda = 0 the density lambda exp(-lambda) is 0 but its
##              slope is 1. nv_dpois() returns 0 there, a documented
##              convention, which the reference records as boundary
##              behaviour.
##   tabulated  3. stirlerr(x) from the table (x <= 15).
##   large      1e6. stirlerr(x) from its series.
##
## Distribution and quantile parameter sets, lambda:
##
##   tiny      1e-30. The upper series' first term lambda / (k + 1) underflows
##             for large enough q, which ppois_raw() takes as its log directly,
##             in both precisions; and the lower series at k = 0, which has no
##             terms but whose scaling overflows for a small enough lambda.
##   small     4. Both series, either side of the mean, with no asymptotic
##             region between them (lambda < 0.8 (k + 51) for every k >= 4).
##   moderate  1e3. The asymptotic expansion between the series: q crosses
##             both seams within ~6 sd of the mean, where the series converge
##             most slowly (term ratios up to 0.8).
##   large     1e6. The asymptotic expansion over +-200 sd of the mean, deep
##             into the tails on the log scale; the series only beyond that.
##
## Each set exists for a code path the others cannot reach, so none is run to
## stand in for another. Negative, zero and infinite lambda, and non-whole x,
## are combinatorial rather than numerical and belong in the unit tests in
## tests/testthat/test-api-distributions.R, not in a 2^32-sample sweep.
##
## base R is the weaker side in three places, each measured against MPFR:
##
##  - dpois() at a lambda with a full mantissa, as the f64 sweep draws them: its
##    dpois_raw() (ebd0()) loses accuracy growing with x, up to ~1.6e4 ulps of
##    the log density and ~2.5e5 of the density at x = 1e6 (~60 and ~260 at
##    x = 1000), where at a round lambda such as 1.025e6 it is within one. At
##    x <= 20 it is within ~9. The density's stable reference below is what lets
##    the sweep tell this from anvl's error.
##  - ppois(lower.tail = FALSE, log.p = TRUE) is -Inf for q from ~2.5e305 at the
##    `small` lambda, where log P(X > q) is still a finite ~-1.7975e308: the
##    deviance term overflows before the whole does. The stable reference below
##    replaces it there.
##  - qpois(log.p = TRUE) is Inf for log p below ~-9e307 (~-9e292 at the `tiny`
##    lambda), where its Cornish-Fisher start overflows; see qpois.R.
##
## Elsewhere its ppois() is pgamma_raw(), whose regions ppois_raw() copies, with
## pgamma_smallx() below lambda = 1, so its upper-tail log does not underflow
## where ppois_raw() needs its own underflow branch. Its probability-scale tails
## carry the usual ~|log P| ulps of a probability formed through exp of its log,
## up to ~1300 ulps near the bottom of the double range.
## ---------------------------------------------------------------------------

POIS_D_PARAMS <- list(
  at_zero = list(x = 0),
  one = list(x = 1),
  tabulated = list(x = 3),
  large = list(x = 1e6)
)

## `tiny` is 1e-30 correctly rounded, written in hex: R's parser reads the
## literal 1e-30 one ulp low on Apple Silicon (0x1.4484bfeebc29fp-100), so the
## decimal would give a different parameter there than on the cluster.
POIS_PQ_PARAMS <- list(
  tiny = list(lambda = 0x1.4484bfeebc2ap-100),
  small = list(lambda = 4),
  moderate = list(lambda = 1e3),
  large = list(lambda = 1e6)
)

## The JAX virtualenv at the root of the sibling checkouts. Anything already
## pinned by RETICULATE_PYTHON wins, so an environment that already carries jax
## is left alone.
jax_init <- local({
  done <- FALSE
  function() {
    if (done) {
      return(invisible(TRUE))
    }
    library(reticulate)
    venv <- normalizePath(file.path(here(), "..", "..", "..", "py-benchmarks", ".venv"), mustWork = FALSE)
    if (!nzchar(Sys.getenv("RETICULATE_PYTHON")) && dir.exists(venv)) {
      use_virtualenv(venv, required = TRUE)
    }
    ## JAX's 64-bit mode is off by default and must be set before any array
    ## exists, or every f64 cell below silently truncates to f32.
    import("jax")$config$update("jax_enable_x64", TRUE)
    done <<- TRUE
    invisible(TRUE)
  }
})

jax_dtype <- function(dtype) {
  jnp <- reticulate::import("jax.numpy", convert = FALSE)
  if (dtype == "f32") jnp$float32 else jnp$float64
}

## An R vector as a JAX array, through NumPy: reticulate hands a bare R vector
## to Python as a list, which JAX then reads value by value -- 0.4 s for a
## chunk of 2^20 samples, against 1 ms. A single value stays a scalar.
jax_array <- function(x, dtype) {
  jnp <- reticulate::import("jax.numpy", convert = FALSE)
  if (length(x) > 1L) {
    x <- reticulate::np_array(x)
  }
  jnp$asarray(x, dtype = jax_dtype(dtype))
}

## jax.scipy.stats.poisson has pmf, logpmf and cdf. logpmf is the direct
## formula xlogy(k, mu) - gammaln(k + 1) - mu, and pmf its exponential. Its cdf
## is gammaincc(floor(k + 1), mu), with no upper tail and no log scale, and
## without base R's floor(q + 1e-7). The distribution function twin is the one
## a JAX user would write from jax.scipy.special, as SciPy's own poisson.cdf
## and poisson.sf do (Cephes pdtr and pdtrc):
##   P(X <= k) = Q(k + 1, lambda),   P(X > k) = P(k + 1, lambda),
## the regularised incomplete gamma functions, with base R's floor(q + 1e-7)
## and its resolution below the support and at q = Inf, so that the two differ
## only in the numerics. JAX differentiates both with respect to lambda, the
## derivative of an incomplete gamma function in its second argument being the
## integrand. There is no quantile function.
##
## Every twin takes lambda at the cell's precision, as anvl does, and the swept
## argument first.
jax_pois <- function() {
  jax_init()
  ## Defined once per process, and each twin jitted once per variant
  ## (lru_cache): rebuilt on every call, JAX re-traced every chunk, at
  ## ~0.5 s a call against ~0.02 s. The program, and so every output, is
  ## the same either way.
  if (!reticulate::py_has_attr(reticulate::py, "_pois_density_value")) {
    reticulate::py_run_string(
      "
import jax
import functools
import jax.numpy as jnp
from jax.scipy.stats import poisson as _po
from jax.scipy.special import gammainc as _gammainc, gammaincc as _gammaincc
@functools.lru_cache(maxsize=None)
def _pois_density_value(log):
    f = _po.logpmf if log else _po.pmf
    return jax.jit(lambda lam, k: f(k, lam))
@functools.lru_cache(maxsize=None)
def _pois_density_grad(log):
    f = _po.logpmf if log else _po.pmf
    g = jax.grad(lambda lam, k: f(k, lam), argnums=0)
    return jax.jit(jax.vmap(g, in_axes=(0, None)))
def _pois_cdf_fn(lower, log):
    def f(q, lam):
        k = jnp.floor(q + 1e-7)
        inside = (k >= 0) & (k < jnp.inf)
        kk = jnp.where(inside, k, 0)
        v = _gammaincc(kk + 1, lam) if lower else _gammainc(kk + 1, lam)
        below, above = (0.0, 1.0) if lower else (1.0, 0.0)
        v = jnp.where(k < 0, below, jnp.where(k == jnp.inf, above, v))
        v = jnp.where(jnp.isnan(q), jnp.nan, v)
        return jnp.log(v) if log else v
    return f
@functools.lru_cache(maxsize=None)
def _pois_cdf(lower, log):
    return jax.jit(_pois_cdf_fn(lower, log))
@functools.lru_cache(maxsize=None)
def _pois_cdf_grad(lower, log):
    g = jax.grad(_pois_cdf_fn(lower, log), argnums=1)
    return jax.jit(jax.vmap(g, in_axes=(0, None)))
"
    )
  }
  invisible(TRUE)
}

jax_dpois_value <- function(lam, dtype, p, log) {
  jax_pois()
  np <- reticulate::import("numpy")
  f <- reticulate::py$`_pois_density_value`(log)
  as.double(np$asarray(f(jax_array(lam, dtype), jax_array(p$x, dtype))))
}

jax_dpois_grad <- function(lam, dtype, p, log) {
  jax_pois()
  np <- reticulate::import("numpy")
  g <- reticulate::py$`_pois_density_grad`(log)
  list(lambda = as.double(np$asarray(g(jax_array(lam, dtype), jax_array(p$x, dtype)))))
}

jax_ppois_value <- function(q, dtype, p, f) {
  jax_pois()
  np <- reticulate::import("numpy")
  cdf <- reticulate::py$`_pois_cdf`(isTRUE(f$lower_tail), isTRUE(f$log_p))
  as.double(np$asarray(cdf(jax_array(q, dtype), jax_array(p$lambda, dtype))))
}

jax_ppois_grad <- function(q, dtype, p, f) {
  jax_pois()
  np <- reticulate::import("numpy")
  g <- reticulate::py$`_pois_cdf_grad`(isTRUE(f$lower_tail), isTRUE(f$log_p))
  list(lambda = as.double(np$asarray(g(jax_array(q, dtype), jax_array(p$lambda, dtype)))))
}

## ---- a stable density -----------------------------------------------------
##
## Loader's saddle point expansion, as base R's dpois_raw() and nv_dpois()
## evaluate it,
##   log p(x) = -stirlerr(x) - bd0(x, lambda) - log(2 pi x) / 2,   x >= 1,
## with the deviance term formed from d = x - lambda, exact where lambda is
## within a factor of two of x and otherwise one rounding of a difference that
## cannot cancel. It is what base R's error at a full-mantissa lambda (see the
## top of this file) is measured against. It shares nv_dpois()'s algorithm, so
## agreement in f64 becomes independent evidence only through the MPFR check
## (REFERENCES.md).

## Stirling's error log(n!) - log(sqrt(2 pi n) (n / e)^n) for whole n >= 1:
## R's exact values up to 15, and the series to 1/n^13 above, which has
## converged to double precision there.
pois_stirlerr <- function(n) {
  table <- c(
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
  nn <- n * n
  series <- (1 /
    12 -
    (1 / 360 - (1 / 1260 - (1 / 1680 - (1 / 1188 - (691 / 360360 - (1 / 156) / nn) / nn) / nn) / nn) / nn) / nn) /
    n
  ifelse(n >= 1 & n <= 15, table[pmax(1, pmin(15, n))], series)
}

## x log(x / m) + m - x for x > 0, m > 0, given d = x - m. Close to m it is
## summed as the series in v = d / (x + m), |v| < 0.1, to v^21: beyond double
## precision. Away from it, x (log(x / m) - 1) + m above m, so that near the
## top of the range the whole is finite wherever the answer is.
pois_bd0 <- function(x, m, d) {
  series <- abs(d) < 0.1 * (x + m)
  v <- d / (x + m)
  w <- v * v
  tail <- 0
  for (j in 10:1) {
    tail <- tail * w + 1 / (2 * j + 1)
  }
  ## x / m overflows where m is tiny beside x -- a lambda of 1e-30 at q above
  ## ~1.8e278 -- and there the difference of the logs, which cannot cancel, is
  ## used instead
  ratio <- x / m
  log_ratio <- ifelse(is.finite(ratio), log(ratio), log(x) - log(m))
  direct <- ifelse(x > m, x * (log_ratio - 1) + m, x * log_ratio + m - x)
  ifelse(series, d * v + 2 * x * v * w * tail, direct)
}

## log dpois(x; lambda) for whole x >= 0 and finite lambda > 0, vectorised over
## x and lambda together.
pois_log_density <- function(x, lam) {
  len <- max(length(x), length(lam))
  x <- rep_len(x, len)
  lam <- rep_len(lam, len)
  out <- -lam
  i <- which(x > 0)
  if (length(i)) {
    k <- x[i]
    l <- lam[i]
    out[i] <- -pois_stirlerr(k) - pois_bd0(k, l, k - l) - 0.5 * (log(2 * pi) + log(k))
  }
  out
}

## The density or its log at lambda, NaN below 0 as base R, with its endpoint
## values at lambda = 0 and Inf.
pois_density_stable <- function(lam, p, f) {
  log <- isTRUE(f$log)
  x <- p$x
  out <- rep(NaN, length(lam))
  ok <- !is.na(lam) & lam >= 0
  at0 <- which(ok & lam == 0)
  out[at0] <- if (x == 0) (if (log) 0 else 1) else (if (log) -Inf else 0)
  out[which(ok & lam == Inf)] <- if (log) -Inf else 0
  i <- which(ok & lam > 0 & lam < Inf)
  if (length(i)) {
    out[i] <- if (log) {
      pois_log_density(x, lam[i])
    } else if (x == 0) {
      exp(-lam[i])
    } else {
      exp(pois_log_density(x, lam[i]))
    }
  }
  out
}

## ---- the tails over the density ---------------------------------------------
##
## The tail series that ppois_raw() also uses, for whole k >= 0 and finite
## lambda > 0:
##
##   P(X > k) / p(k)  = (lambda / (k + 1)) (1 + lambda / (k + 2)
##                        + lambda^2 / ((k + 2)(k + 3)) + ...)
##   P(X <= k) / p(k) = 1 + k / lambda + k (k - 1) / lambda^2 + ...  (k terms)
##
## summed until a term falls below 2^-60 of the total. They converge
## geometrically wherever the first term ratio r (lambda / (k + 1), k / lambda)
## is below one; below 0.99 that is at most ~4000 terms. Each term is a running
## product, so its rounding grows with its index, and the sum's relative error
## is bounded by about twice the mean index, 2 r / (1 - r) ulps, though the
## roundings mostly cancel. Returns the sum without its leading factor, that is
## P(X > k) / p(k) over lambda / (k + 1) for the upper tail.
pois_tail_sum <- function(k, lam, lower) {
  step <- if (lower) function(j) pmax(k - j + 1, 0) / lam else function(j) lam / (k + 1 + j)
  term <- rep(1, length(k))
  total <- term
  for (j in seq_len(100000)) {
    term <- term * step(j)
    total <- total + term
    if (all(term < 2^-60 * total)) {
      break
    }
  }
  total
}

## p(k) / P(X <= k) (lower) or p(k) / P(X > k) (upper), for the log-scale
## gradients. By the series wherever r < 0.99. Elsewhere, near the mean, the
## quotient of base R's values, whose tails are not small there; base R's tails
## carry ~|log P| ulps (see the top of this file), which a quotient taken in the
## far tails would inherit. Where either side of the quotient has underflowed
## and r >= 0.99, which no set reaches (an underflowing tail needs
## lambda > ~1.4e7 there), it falls back to the difference of base R's logs.
pois_density_over_tail <- function(k, lam, lower) {
  first <- if (lower) k / lam else lam / (k + 1)
  out <- rep(NaN, length(k))
  s <- which(first < 0.99)
  if (length(s)) {
    ks <- k[s]
    total <- pois_tail_sum(ks, lam, lower)
    out[s] <- if (lower) {
      1 / total
    } else {
      ## (k + 1) / lambda overflows only where the answer itself is within a
      ## factor of total (< 100) of doing so
      lead <- (ks + 1) / lam
      ifelse(is.finite(lead), lead / total, exp(log(ks + 1) - log(lam) - log(total)))
    }
  }
  near <- setdiff(seq_along(k), s)
  if (length(near)) {
    kn <- k[near]
    d <- dpois(kn, lam)
    tail <- ppois(kn, lam, lower.tail = lower)
    out[near] <- ifelse(
      d >= 2^-1022 & tail >= 2^-1022,
      d / tail,
      exp(dpois(kn, lam, log = TRUE) - ppois(kn, lam, lower.tail = lower, log.p = TRUE))
    )
  }
  out
}

## ---- a stable upper-tail log distribution function --------------------------
##
## log P(X > k), k = floor(q + 1e-7), for the upper-tail log cells. base R's
## ppois() returns -Inf for q from ~2.5e305 at the `small` lambda, where the
## answer is a finite ~-1.7975e308: the deviance term it forms overflows before
## the whole does. That is the failure this reference exists to recognise, and
## it replaces base R only there: where base R is -Inf at a finite q whose
## series converges (r < 0.9), it is the stable log density, log(lambda) -
## log(k + 1) and the log of pois_tail_sum(), in [0, 10]. Elsewhere it is base
## R's own value, whose error the declared bound covers, so that a dispute can
## only arise where base R has failed outright.
pois_log_upper_stable <- function(q, lam) {
  out <- ppois(q, lam, lower.tail = FALSE, log.p = TRUE)
  k <- floor(q + 1e-7)
  far <- which(!is.na(q) & q >= 0 & q < Inf & out == -Inf & lam / (k + 1) < 0.9)
  if (length(far)) {
    kf <- k[far]
    r <- lam / (kf + 1)
    log_r <- ifelse(r >= 2^-1022, log(r), log(lam) - log(kf + 1))
    out[far] <- pois_log_density(kf, lam) + log_r + log(pois_tail_sum(kf, lam, FALSE))
  }
  out
}

## ---- high-precision truths, for validate-refs only -------------------------
##
## Rmpfr is needed here and nowhere else; a sweep never calls these. The
## parameters arrive as mpfr numbers, exact.

## The Poisson distribution at whole k >= 0 (a double vector, as base R takes
## floor(q + 1e-7)) and finite lambda > 0, in MPFR, by the regularised
## incomplete gamma functions of a = k + 1 at lambda: P(X > k) = P(a, lambda)
## and P(X <= k) = Q(a, lambda). Where lambda < a + 1 by the series
##   P(a, x) = x^a e^-x / Gamma(a + 1) (1 + x / (a + 1) + x^2 / ((a + 1)(a + 2)) + ...),
## and otherwise by Lentz's continued fraction for Q(a, x), so each is taken
## on its convergent side (Numerical Recipes' gser and gcf). Both come out as a
## tail over p(k), with no exponential of log p(k): at k = 1e300 that log is
## ~7e302, and at 256 bits its absolute error is ~1e225. Each element iterates
## until its last step changes it by less than 2^(10 - precision). None of it
## shares the double references' methods.
##
## Returns log p(k), the log of each tail, and `ratio`, the tail over p(k) that
## was summed: P(X > k) / p(k) where `series`, P(X <= k) / p(k) elsewhere.
mp_pois_parts <- function(k, lambda) {
  prec <- mp_prec(lambda)
  kk <- Rmpfr::mpfr(k, prec)
  a <- kk + 1
  log_density <- kk * log(lambda) - lambda - lgamma(kk + 1)
  series <- mp_num(lambda) < k + 2
  tol <- 2^(10 - prec)
  ratio <- mp_fill(kk, NaN)

  s <- which(series)
  if (length(s)) {
    as_ <- a[s]
    term <- mp_fill(as_, 1)
    total <- term
    live <- seq_along(s)
    for (n in seq_len(1000000)) {
      term[live] <- term[live] * lambda / (as_[live] + n)
      total[live] <- total[live] + term[live]
      live <- live[mp_num(term[live] / total[live]) >= tol]
      if (!length(live)) {
        break
      }
    }
    ratio[s] <- lambda / as_ * total
  }

  cf <- which(!series)
  if (length(cf)) {
    ac <- a[cf]
    b <- lambda + 1 - ac
    cc <- mp_fill(ac, 2^1000)
    d <- 1 / b
    h <- d
    live <- seq_along(cf)
    for (i in seq_len(1000000)) {
      an <- -i * (i - ac[live])
      b[live] <- b[live] + 2
      d[live] <- an * d[live] + b[live]
      cc[live] <- b[live] + an / cc[live]
      d[live] <- 1 / d[live]
      del <- d[live] * cc[live]
      h[live] <- h[live] * del
      live <- live[abs(mp_num(del - 1)) >= tol]
      if (!length(live)) {
        break
      }
    }
    ratio[cf] <- lambda * h
  }

  log_upper <- mp_fill(kk, NaN)
  log_lower <- mp_fill(kk, NaN)
  if (length(s)) {
    log_upper[s] <- log_density[s] + log(ratio[s])
    log_lower[s] <- log1p(-exp(log_upper[s]))
  }
  if (length(cf)) {
    log_lower[cf] <- log_density[cf] + log(ratio[cf])
    log_upper[cf] <- log1p(-exp(log_lower[cf]))
  }
  list(log_density = log_density, log_lower = log_lower, log_upper = log_upper, ratio = ratio, series = series)
}

## ---- where nv_ppois() switches algorithm -------------------------------------
##
## ppois_raw() (R/api-distributions.R) takes the upper series where
## lambda <= k and lambda < 0.8 (k + 51), the lower series where k < lambda and
## k + 1 < 0.8 (lambda + 50), and the asymptotic expansion between them, at
## k = floor(q + 1e-7). Each region is a set of whole numbers, so a switch is
## between two neighbouring counts; the point returned is the larger, as a q.
## They are found by testing the conditions at candidate counts around their
## solutions, in double: anvl tests them at the cell's precision, so in f32 a
## switch can move by a count where a condition is within rounding of an
## integer.
pois_cdf_switches <- function(lambda) {
  region <- function(k) {
    upper <- (lambda <= k) & (lambda < 0.8 * (k + 51))
    lower <- (k < lambda) & (k + 1 < 0.8 * (lambda + 50))
    ifelse(upper, "upper", ifelse(lower, "lower", "asymptotic"))
  }
  est <- c(lambda, 1.25 * lambda - 51, 0.8 * lambda + 39)
  ks <- sort(unique(as.vector(outer(floor(est), -2:3, `+`))))
  ks <- ks[ks >= 1]
  sw <- ks[region(ks) != region(ks - 1)]
  setNames(sw, paste0(region(sw - 1), "_to_", region(sw), recycle0 = TRUE))
}
