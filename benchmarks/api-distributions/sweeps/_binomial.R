## ---------------------------------------------------------------------------
## Shared by the three binomial specs. Not a spec itself (leading underscore).
##
## The density is swept in `prob`, not in x. A bit-pattern sweep of x lands on
## an integer only from 2^23 (f32) or 2^52 (f64) upwards, so for any ordinary
## size it would test almost nothing but the zero density at non-whole x.
## Swept in prob, with x and size fixed, the density is a smooth function on
## [0, 1], its gradient with respect to prob is the one anvl differentiates,
## and n p sweeps past x, crossing the seams of Loader's deviance terms. The
## distribution and quantile functions are swept in q and p as usual.
##
## Density parameter sets, (x, size):
##
##   tabulated     (3, 10). Every Stirling correction comes from the table
##                 (n, x and n - x all <= 15), and log(1 - x / n) is taken by
##                 log1p.
##   next_to_zero  (1, 10). At prob = 0 the density is 0 but its slope is
##                 size, which nv_dbinom() keeps by writing the density out as
##                 n p (1 - p)^(n - 1) there.
##   next_to_size  (9, 10). The same at prob = 1, and x > n / 2, so
##                 log(1 - x / n) is taken as log(n - x) - log(n).
##   at_zero       (0, 1000). The closed form (1 - p)^n.
##   at_size       (1000, 1000). The closed form p^n.
##   large         (3e5, 1e6). Every Stirling correction from its series, and
##                 the deviance terms both ways round. x - n p is carried
##                 exactly, which base R's rounded n p loses near the mode
##                 (~1e-13). JAX's logpmf differences three lgamma values of
##                 ~1e7, which loses ~1e-9.
##   huge          (3e12, 1e13). As `large`, where it matters most: base R is
##                 ~1e-9 off and JAX's lgamma difference ~1e-2.
##
## Distribution and quantile parameter sets, (size, prob):
##
##   small   (20, 0.3). The continued fraction of TOMS 708 in both
##           orientations, either side of the mean, with tabulated Stirling
##           corrections in its leading factor.
##   large   (1e6, 0.3). The asymptotic expansion near the mean, where
##           min(a, b) > 1000 and lambda <= 0.03 min(a, b), and the continued
##           fraction in both tails: q crosses both seams.
##   skewed  (1e6, 1e-3). n p = 1000, so near the mean min(a, b) itself
##           crosses 1000: the expansion's other condition. JAX's betainc
##           needs more than its 600 continued-fraction iterations near the
##           mean of `large`; here it does not.
##   rare    (1e10, 1e-10). n p = 1, so a = k + 1 is tiny beside b = n - k:
##           the regime in which the continued fraction's scaling must keep
##           every term representable (the regression of the first
##           division-free rewrite).
##
## Each set exists for a code path the others cannot reach, so none is run to
## stand in for another. Zero, negative, non-whole and infinite sizes, prob
## outside [0, 1], and non-whole x are combinatorial rather than numerical and
## belong in the unit tests in tests/testthat/test-api-distributions.R, not in
## a 2^32-sample sweep.
## ---------------------------------------------------------------------------

BINOM_D_PARAMS <- list(
  tabulated = list(x = 3, size = 10),
  next_to_zero = list(x = 1, size = 10),
  next_to_size = list(x = 9, size = 10),
  at_zero = list(x = 0, size = 1000),
  at_size = list(x = 1000, size = 1000),
  large = list(x = 3e5, size = 1e6),
  huge = list(x = 3e12, size = 1e13)
)

BINOM_PQ_PARAMS <- list(
  small = list(size = 20, prob = 0.3),
  large = list(size = 1e6, prob = 0.3),
  skewed = list(size = 1e6, prob = 1e-3),
  rare = list(size = 1e10, prob = 1e-10)
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

## jax.scipy.stats.binom has pmf and logpmf only, by the direct formula
## lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k log p + (n - k) log1p(-p).
## It has no distribution or quantile function. The distribution function
## twin is the one a JAX user would write from jax.scipy.special.betainc, as
## SciPy's own binom.cdf and binom.sf do (Cephes bdtr and bdtrc):
##   P(X <= k) = I_(1 - p)(n - k, k + 1),   P(X > k) = I_p(k + 1, n - k),
## with base R's floor(q + 1e-7) and its resolution below and above the
## support, so that the two differ only in the numerics. JAX's betainc is a
## modified Lentz continued fraction capped at 200 (f32) or 600 (f64)
## iterations, with its leading factor from lgamma. Its gradient in a and b is
## not implemented, and anvl's distribution function has none either.
##
## Every twin takes size and prob at the cell's precision, as anvl does, and
## the swept argument first.
jax_binom <- function() {
  jax_init()
  ## Defined once per process, and each twin jitted once per variant
  ## (lru_cache): rebuilt on every call, JAX re-traced every chunk, at
  ## ~0.5 s a call against ~0.02 s. The program, and so every output, is
  ## the same either way.
  if (!reticulate::py_has_attr(reticulate::py, "_binom_density_value")) {
    reticulate::py_run_string(
      "
import jax
import functools
import jax.numpy as jnp
from jax.scipy.stats import binom as _b
from jax.scipy.special import betainc as _betainc
@functools.lru_cache(maxsize=None)
def _binom_density_value(log):
    f = _b.logpmf if log else _b.pmf
    return jax.jit(lambda p, k, n: f(k, n, p))
@functools.lru_cache(maxsize=None)
def _binom_density_grad(log):
    f = _b.logpmf if log else _b.pmf
    g = jax.grad(lambda p, k, n: f(k, n, p), argnums=0)
    return jax.jit(jax.vmap(g, in_axes=(0, None, None)))
@functools.lru_cache(maxsize=None)
def _binom_cdf(lower, log):
    def f(q, n, p):
        k = jnp.floor(q + 1e-7)
        inside = (k >= 0) & (k < n)
        kk = jnp.where(inside, k, 0)
        if lower:
            v = _betainc(n - kk, kk + 1, 1 - p)
        else:
            v = _betainc(kk + 1, n - kk, p)
        below, above = (0.0, 1.0) if lower else (1.0, 0.0)
        v = jnp.where(k < 0, below, jnp.where(k >= n, above, v))
        v = jnp.where(jnp.isnan(q), jnp.nan, v)
        return jnp.log(v) if log else v
    return jax.jit(f)
"
    )
  }
  invisible(TRUE)
}

jax_scalar <- function(v, dtype) jax_array(v, dtype)

jax_dbinom_value <- function(pr, dtype, p, log) {
  jax_binom()
  np <- reticulate::import("numpy")
  f <- reticulate::py$`_binom_density_value`(log)
  as.double(np$asarray(f(jax_scalar(pr, dtype), jax_scalar(p$x, dtype), jax_scalar(p$size, dtype))))
}

jax_dbinom_grad <- function(pr, dtype, p, log) {
  jax_binom()
  np <- reticulate::import("numpy")
  g <- reticulate::py$`_binom_density_grad`(log)
  list(prob = as.double(np$asarray(g(jax_scalar(pr, dtype), jax_scalar(p$x, dtype), jax_scalar(p$size, dtype)))))
}

jax_pbinom_value <- function(q, dtype, p, f) {
  jax_binom()
  np <- reticulate::import("numpy")
  cdf <- reticulate::py$`_binom_cdf`(isTRUE(f$lower_tail), isTRUE(f$log_p))
  as.double(np$asarray(cdf(jax_scalar(q, dtype), jax_scalar(p$size, dtype), jax_scalar(p$prob, dtype))))
}

## ---- a stable density -----------------------------------------------------
##
## Loader's saddle point expansion, as base R's dbinom_raw() and nv_dbinom()
## evaluate it, with two differences from base R that are its whole point:
##
##  - d = x - n p is formed from the exact product n p = hi + lo (Dekker's
##    TwoProduct), not from a rounded n p. Rounding n p costs the deviance
##    terms about d ulp(n p) / (n p), which is base R's error at large size.
##  - log(1 - x / n) is log(n - x) - log(n) for x > n / 2, where n - x is
##    exact and x / n is not.
##
## It shares nv_dbinom()'s algorithm, so agreement in f64 becomes independent
## evidence only through the MPFR check (REFERENCES.md).

## n * p as hi + lo, exact. Outside 1e-290 < |hi| < 1e290 the split would
## under- or overflow and lo is taken as 0; d is then dominated by hi.
binom_np <- function(n, p) {
  hi <- n * p
  lo <- numeric(length(hi))
  split <- function(a) {
    c <- 134217729 * a
    h <- c - (c - a)
    list(h = h, l = a - h)
  }
  ok <- which(is.finite(hi) & abs(hi) > 1e-290 & abs(hi) < 1e290)
  if (length(ok)) {
    a <- split(rep(n, length(ok)))
    b <- split(p[ok])
    lo[ok] <- ((a$h * b$h - hi[ok]) + a$h * b$l + a$l * b$h) + a$l * b$l
  }
  list(hi = hi, lo = lo)
}

## Stirling's error log(n!) - log(sqrt(2 pi n) (n / e)^n) for whole n >= 1:
## R's exact values up to 15, and the series to 1/n^13 above, which has
## converged to double precision there.
binom_stirlerr <- function(n) {
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

## x log(x / m) + m - x for x > 0, m > 0, given d = x - m exactly. Close to m
## it is summed as the series in v = d / (x + m), |v| < 0.1, to v^21: beyond
## double precision.
binom_bd0 <- function(x, m, d) {
  series <- abs(d) < 0.1 * (x + m)
  v <- d / (x + m)
  w <- v * v
  tail <- 0
  for (j in 10:1) {
    tail <- tail * w + 1 / (2 * j + 1)
  }
  ## x / m overflows where m is tiny -- a subnormal n p, which a double keeps
  ## (XLA flushes it, so nv_dbinom() needs no such case) -- and there the
  ## difference of the logs, which cannot cancel, is used instead
  ratio <- x / m
  log_ratio <- ifelse(is.finite(ratio), log(ratio), log(x) - log(m))
  direct <- ifelse(x > m, x * (log_ratio - 1) + m, x * log_ratio + m - x)
  ifelse(series, d * v + 2 * x * v * w * tail, direct)
}

## log dbinom(x; n, p) for whole 0 <= x <= n and p in [0, 1], vectorised over
## x and p together.
binom_log_density <- function(pr, x, n) {
  len <- max(length(pr), length(x))
  pr <- rep_len(pr, len)
  x <- rep_len(x, len)
  out <- rep(-Inf, len)
  at0 <- which(x == 0)
  out[at0] <- n * log1p(-pr[at0])
  atn <- which(x == n)
  out[atn] <- n * log(pr[atn])
  i <- which(x > 0 & x < n & pr > 0 & pr < 1)
  if (length(i)) {
    q <- pr[i]
    k <- x[i]
    np <- binom_np(n, q)
    d <- (k - np$hi) - np$lo
    lc <- binom_stirlerr(n) -
      binom_stirlerr(k) -
      binom_stirlerr(n - k) -
      binom_bd0(k, np$hi, d) -
      binom_bd0(n - k, n * (1 - q), -d)
    log_frac <- ifelse(k > n / 2, log(n - k) - log(n), log1p(-k / n))
    out[i] <- lc - 0.5 * (log(2 * pi) + log(k) + log_frac)
  }
  out
}

## The density at x = 0 or x = n without exponentiating its log, whose
## rounding exp() would multiply by |n log|: p^n directly, p being exact, and
## (1 - p)^n directly where 1 - p is exact, as base R's pow1p() does; only
## where it is not, exp(n log1p(-p)).
binom_end_density <- function(pr, x, n) {
  if (x == n) {
    return(pr^n)
  }
  q <- 1 - pr
  ifelse((1 - q) == pr, q^n, exp(n * log1p(-pr)))
}

## The density or its log at p, NaN outside [0, 1] as base R.
binom_density_stable <- function(pr, p, f) {
  out <- rep(NaN, length(pr))
  ok <- which(!is.na(pr) & pr >= 0 & pr <= 1)
  if (length(ok)) {
    out[ok] <- if (isTRUE(f$log)) {
      binom_log_density(pr[ok], p$x, p$size)
    } else if (p$x == 0 || p$x == p$size) {
      binom_end_density(pr[ok], p$x, p$size)
    } else {
      exp(binom_log_density(pr[ok], p$x, p$size))
    }
  }
  out
}

## ---- a stable lower-tail log distribution function -----------------------
##
## log P(X <= k), k = floor(q + 1e-7), for the lower-tail log cells. base R's
## pbeta() fails in the far lower tail: it underflows to -Inf (-Inf at q = 25
## for size 1e6, prob 0.3, where log P = -3.6e5), or returns a finite but wrong
## value (-603 at q = 37.5 for size 1e6, prob 1e-3, where log P = -844), or, at
## sizes beyond these sets, an impossible positive one. That is the failure
## this reference exists to recognise, and it replaces base R only in that
## tail. Elsewhere it is base R's own value, whose error the declared bound
## covers, so that a dispute can only arise where base R has failed outright.
##
## In the far lower tail the binomial sum falls away geometrically:
##   P(X <= k) = dbinom(k) (1 + r_k + r_k r_(k-1) + ...),
##   r_j = j (1 - p) / ((n - j + 1) p),
## the ratio of the mass at j - 1 to that at j, below 1 below the mode and
## shrinking away from it. It is taken where r_k < 0.9, and wherever base R's
## log is below -600 or not finite; the sum runs until a term falls below
## 2^-60 of the total, with the stable log density at k, whose log is the size
## of log P there, so its rounding costs only a few ulps of the result.
binom_log_cdf_lower <- function(q, n, pr) {
  out <- suppressWarnings(pbinom(q, n, pr, log.p = TRUE))
  k <- floor(q + 1e-7)
  r_k <- ifelse(k >= 1, k * (1 - pr) / ((n - k + 1) * pr), 0)
  far <- which(!is.na(q) & q >= 0 & k < n & (r_k < 0.9 | is.na(out) | out <= -600))
  if (length(far)) {
    kk <- k[far]
    j <- kk
    term <- rep(1, length(far))
    total <- term
    for (m in seq_len(100000)) {
      ratio <- ifelse(j >= 1, j * (1 - pr) / ((n - j + 1) * pr), 0)
      term <- term * ratio
      total <- total + term
      j <- j - 1
      if (all(term < 2^-60 * total)) {
        break
      }
    }
    out[far] <- binom_log_density(pr, kk, n) + log(total)
  }
  out
}

## ---- high-precision truths, for validate-refs only -------------------------
##
## Rmpfr is needed here and nowhere else; a sweep never calls these. The
## parameters arrive as mpfr numbers, exact.

## log dbinom(x; n, p) in MPFR, by lgamma: at 256 bits lgamma(1e13) ~ 2.9e14
## still carries ~180 bits below the point.
mp_binom_log_density <- function(pr, x, n) {
  xn <- mp_num(x)
  nn <- mp_num(n)
  out <- lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1) + 0 * pr
  if (xn > 0) {
    out <- out + x * log(pr)
  }
  if (xn < nn) {
    out <- out + (n - x) * log1p(-pr)
  }
  out
}

## ---- where nv_pbinom() switches algorithm ------------------------------------
##
## binom_cdf() (R/api-distributions.R) evaluates I_x(a, b) at k = floor(q +
## 1e-7) on the side where lambda = (k + 1) - (n + 1) p >= 0 -- P(X > k)
## directly there, P(X <= k) below -- and by the asymptotic expansion where
## min(a, b) > 1000 and |lambda| <= 0.03 min(a, b), the continued fraction
## elsewhere. Each is a function of the whole number k, so it switches between
## two neighbouring counts; the point returned is the larger, as a q. They are
## found by testing the conditions at candidate counts around their solutions,
## in double: anvl forms lambda at the cell's precision, so in f32 a switch can
## move by a count where the condition is within rounding of an integer.
binom_cdf_switches <- function(n, p) {
  lambda <- function(k) (k + 1) - (n + 1) * p
  asym <- function(k) {
    l <- lambda(k)
    up <- l >= 0
    m <- pmin(ifelse(up, k + 1, n - k), ifelse(up, n - k, k + 1))
    m > 1000 & abs(l) <= 0.03 * m
  }
  np1 <- (n + 1) * p
  est <- c(
    np1 - 1,
    np1 / 0.97 - 1,
    (np1 - 1 + 0.03 * n) / 1.03,
    np1 / 1.03 - 1,
    (np1 - 1 - 0.03 * n) / 0.97,
    999,
    1000,
    n - 1001,
    n - 1000
  )
  ks <- sort(unique(as.vector(outer(floor(est), -2:3, `+`))))
  ks <- ks[ks >= 1 & ks <= n - 1]
  side <- ks[(lambda(ks) >= 0) != (lambda(ks - 1) >= 0)]
  expansion <- ks[asym(ks) != asym(ks - 1)]
  c(
    setNames(side, rep("lambda_sign", length(side))),
    setNames(expansion, paste0("asymptotic_", seq_along(expansion), recycle0 = TRUE))
  )
}

## log P(X <= k), k = floor(q + 1e-7) in double as base R takes it, in MPFR,
## by the incomplete beta function P(X <= k) = I_(1 - p)(n - k, k + 1):
## Lentz's continued fraction for I_x(a, b) where x < (a + 1) / (a + b + 2),
## and 1 - I_p(k + 1, n - k) otherwise, so that the fraction is always taken on
## its convergent side and the complement is of a tail no larger than about a
## half. The leading factor x^a (1 - x)^b / (a B(a, b)) is formed from lgamma.
## The iteration runs until every sample's last step changes it by less than
## 2^(10 - precision); none of it shares the double reference's method.
mp_log_pbinom_lower <- function(q, n, p) {
  prec <- mp_prec(q)
  qn <- mp_num(q)
  nn <- mp_num(n)
  k <- floor(qn + 1e-7)
  out <- mp_fill(q, NaN)
  out[!is.na(qn) & qn < 0] <- -Inf
  out[!is.na(qn) & qn >= 0 & k >= nn] <- 0
  i <- which(!is.na(qn) & qn >= 0 & k < nn)
  if (!length(i)) {
    return(out)
  }
  kk <- Rmpfr::mpfr(k[i], prec)
  one <- Rmpfr::mpfr(1, prec)
  y <- one - p
  a0 <- n - kk
  b0 <- kk + 1
  direct <- mp_num(y) < mp_num((a0 + 1) / (a0 + b0 + 2))
  a <- ifelse(direct, mp_num(a0), mp_num(b0))
  a <- Rmpfr::mpfr(a, prec)
  b <- Rmpfr::mpfr(ifelse(direct, mp_num(b0), mp_num(a0)), prec)
  x <- y + 0 * a
  x[!direct] <- p
  qab <- a + b
  qap <- a + 1
  qam <- a - 1
  cc <- one + 0 * a
  d <- 1 / (1 - qab * x / qap)
  h <- d
  tol <- 2^(10 - prec)
  for (m in seq_len(200000)) {
    m2 <- 2 * m
    aa <- m * (b - m) * x / ((qam + m2) * (a + m2))
    d <- 1 / (1 + aa * d)
    cc <- 1 + aa / cc
    h <- h * d * cc
    aa <- -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
    d <- 1 / (1 + aa * d)
    cc <- 1 + aa / cc
    del <- d * cc
    h <- h * del
    if (all(abs(mp_num(del - 1)) < tol)) {
      break
    }
  }
  log_i <- a * log(x) + b * log(1 - x) - (lgamma(a) + lgamma(b) - lgamma(qab)) - log(a) + log(h)
  res <- log_i
  res[!direct] <- log1p(-exp(log_i[!direct]))
  out[i] <- res
  out
}
