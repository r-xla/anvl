## ---------------------------------------------------------------------------
## Shared by the three exponential specs. Not a spec itself (leading underscore).
##
##   standard  rate = 1, the default. Every multiplication by the rate is exact,
##             so it isolates the distribution's own mathematics -- and says
##             nothing about the scaling.
##   small     rate = pi * 1e-10. Representable in neither precision, and small
##             enough that rate * q underflows for normal q (q below 7e-299 in
##             f64, 3.7e-29 in f32): the only route into nv_pexp's rescaled
##             log_p branch. It also reaches nv_qexp's split exponential, where
##             exp(p) underflows but exp(p) / rate is still representable.
##   large     rate = pi * 1e10. Representable in neither precision, and large
##             enough that rate * exp(-t) is representable where exp(-t) has
##             already underflowed: the only route into the part of nv_dexp's
##             split exponential that changes the answer (t in (708.4, 732.6)
##             in f64, (87.3, 111.5) in f32).
##
## Each set exists for a code path the other two cannot reach, so none is run
## to stand in for another. Zero, infinite and negative rates are
## combinatorial rather than numerical and belong in the unit tests in
## tests/testthat/test-api-distributions.R, not in a 2^32-sample sweep.
## ---------------------------------------------------------------------------

EXP_PARAMS <- list(
  standard = list(rate = 1),
  small = list(rate = pi * 1e-10),
  large = list(rate = pi * 1e10)
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

## jax.scipy.stats.expon is parameterised by scale = 1/rate. Every JAX twin
## takes the rate and forms the scale inside the traced function, as a JAX user
## holding a rate would, so a gradient with respect to its second argument is
## d/drate, comparable with anvl's, not d/dscale.
jax_expon <- function() {
  jax_init()
  reticulate::py_run_string(
    "
import jax
from jax.scipy.stats import expon as _e
_EXPON = {
    'pdf': _e.pdf, 'logpdf': _e.logpdf,
    'cdf': _e.cdf, 'logcdf': _e.logcdf, 'sf': _e.sf, 'logsf': _e.logsf,
    'ppf': _e.ppf,
}
def _expon_value(name):
    f = _EXPON[name]
    return jax.jit(lambda x, r: f(x, scale=1 / r))
def _expon_grad(name, i):
    f = _EXPON[name]
    g = jax.grad(lambda x, r: f(x, scale=1 / r), argnums=i)
    return jax.jit(jax.vmap(g, in_axes=(0, None)))
"
  )
  invisible(TRUE)
}

jax_expon_value <- function(name, x, dtype, rate) {
  jax_expon()
  jnp <- reticulate::import("jax.numpy", convert = FALSE)
  np <- reticulate::import("numpy")
  as.double(np$asarray(reticulate::py$`_expon_value`(name)(jnp$asarray(x, dtype = jax_dtype(dtype)), rate)))
}

jax_expon_grad <- function(name, x, dtype, rate, wrt) {
  jax_expon()
  jnp <- reticulate::import("jax.numpy", convert = FALSE)
  np <- reticulate::import("numpy")
  xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
  g <- function(i) as.double(np$asarray(reticulate::py$`_expon_grad`(name, i)(xx, rate)))
  setNames(list(g(0L), g(1L)), wrt)
}

## t = rate * x as an unevaluated sum hi + lo (Dekker's TwoProduct), exact.
##
## A double t is off by up to half an ulp, and exp(-t) multiplies that by t:
## ~350 ulp at t = 700. anvl and base R both round t, so a reference that did
## too would share their conditioning error instead of measuring it. In an f32
## cell both factors carry 24 bits and the product is exact in double, so lo is
## exactly 0 there. Outside 1e-290 < |hi| < 1e290 the split would under- or
## overflow and lo is taken as 0: below, every use of t is insensitive to its
## last ulp; above, exp(-t) is 0.
rate_times <- function(x, rate) {
  hi <- rate * x
  lo <- numeric(length(hi))
  split <- function(a) {
    c <- 134217729 * a
    h <- c - (c - a)
    list(h = h, l = a - h)
  }
  ok <- which(is.finite(hi) & abs(hi) > 1e-290 & abs(hi) < 1e290)
  if (length(ok)) {
    a <- split(x[ok])
    b <- split(rep(rate, length(ok)))
    lo[ok] <- ((a$h * b$h - hi[ok]) + a$h * b$l + a$l * b$h) + a$l * b$l
  }
  list(hi = hi, lo = lo)
}

## m * exp(-(t + lo)) for t >= 0, without forming exp(-t) first.
##
## exp(-t) underflows at t ~ 708.4 (745 once subnormal) while m * exp(-t) can
## still be representable -- rate^2 exp(-t) at the `large` rate is ~1e-301 at
## t = 740. Past t = 700 it is (m exp(-t/2)) exp(-t/2), t/2 exact, so nothing
## underflows before the answer does. Past t = 1500 it is a zero signed like m
## for any m a double can hold (|m| e^-1500 < 2^-1075), including where t or m
## has overflowed; this is also where lo stops being small in absolute terms
## (ulp(t) > 1 from t = 2^53), so exp(-lo) is folded in only below it, and
## only where lo is non-zero. m is a factor of ordinary size (rate, rate^2, q,
## 1 - t), never subnormal.
exp_times <- function(t, m, lo = 0) {
  lo <- rep_len(lo, length(t))
  m <- rep_len(m, length(t))
  j <- which(lo != 0 & t <= 1500)
  if (length(j)) {
    m[j] <- m[j] * exp(-lo[j])
  }
  out <- m * exp(-t)
  far <- which(!is.na(t) & t > 700 & t <= 1500)
  if (length(far)) {
    h <- exp(-t[far] / 2)
    out[far] <- (m[far] * h) * h
  }
  gone <- which(!is.na(t) & t > 1500)
  out[gone] <- 0 * sign(m[gone])
  out
}

## exp(lp) / rate for lp <= 0, without exp(lp) underflowing first.
##
## Below lp = -700, exp(lp) is near or below the smallest normal while its
## quotient by a small rate may be ordinary: (exp(lp + 700) / rate) * exp(-700),
## with lp + 700 exact (Sterbenz) for lp >= -1400. Below that the quotient is
## under half the smallest subnormal for every rate in EXP_PARAMS (it would
## need rate < e^-655), so it is 0.
exp_over <- function(lp, rate) {
  out <- exp(lp) / rate
  far <- which(!is.na(lp) & lp < -700)
  if (length(far)) {
    out[far] <- ifelse(lp[far] >= -1400, (exp(lp[far] + 700) / rate) * exp(-700), 0)
  }
  out
}

## log(1 - exp(-(t + lo))) for t > 0: the lower-tail log CDF at t = rate * q.
##
##   t below the smallest normal  log(rate) + log(q). Where t is subnormal or
##                                zero in double the product has lost its bits,
##                                but its two factors have not; the logs share
##                                a sign or one is small, so they do not cancel.
##   t <= log 2                   log(-expm1(-t)), with lo folded in to first
##                                order: + lo / expm1(t).
##   t > log 2                    log1p(-exp(-t) exp(-lo)), lo folded in only
##                                below t = 1500, past which exp(-t) is 0 and
##                                lo is no longer small (see exp_times()).
##
## The two forms either side of log 2 are Maechler's log1mexp split, which is
## also what nv_pexp() uses; the reference is evaluated in double, 29 bits
## beyond an f32 result, but for f64 cells it shares anvl's formula -- which is
## why its bound is checked against high precision rather than trusted.
log_cdf_lower <- function(q, rate) {
  tt <- rate_times(q, rate)
  t <- tt$hi
  lo <- tt$lo
  out <- rep(NaN, length(q))
  tiny <- which(t < 2^-1022)
  out[tiny] <- log(rate) + log(q[tiny])
  small <- which(t >= 2^-1022 & t <= log(2))
  out[small] <- log(-expm1(-t[small])) + lo[small] / expm1(t[small])
  big <- which(t > log(2))
  out[big] <- log1p(-exp(-t[big]) * ifelse(t[big] <= 1500, exp(-lo[big]), 1))
  out
}

## log(1 - exp(lp)) for lp <= 0, the same split, on a log probability.
log1mexp_ref <- function(lp) ifelse(lp > -log(2), log(-expm1(lp)), log1p(-exp(lp)))

## ---- high-precision truths, for validate-refs only -------------------------
##
## Rmpfr is needed here and nowhere else; a sweep never calls these. Each
## evaluates the mathematics in MPFR at the precision of its arguments, and
## conditions are tested on the exact doubles, which are exact.

## log(1 - exp(v)) for v <= 0 in MPFR. Even at 256 bits 1 - exp(v) rounds to
## 1 below v ~ -177, so the split is needed here too, not only in double.
mp_log1mexp <- function(v) {
  vn <- mp_num(v)
  out <- v
  near <- which(!is.na(vn) & vn > -log(2))
  if (length(near)) {
    out[near] <- log(-expm1(v[near]))
  }
  far <- which(!is.na(vn) & vn <= -log(2))
  if (length(far)) {
    out[far] <- log1p(-exp(v[far]))
  }
  out
}
