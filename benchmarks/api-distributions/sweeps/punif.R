## ---------------------------------------------------------------------------
## nv_punif against base R punif().
##
## The log_p variants are the interesting ones: that is where nv_punif()
## switches from log(u) to log1p(-v) of the opposite tail at u = 1/2, and the
## sweep crosses that seam from both sides.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_uniform.R"), local = TRUE)

## Write w = max - min > 0. On the strict interior min < q < max the lower-tail
## CDF is F = (q - min)/w, so
##
##   d/dq  1/w      d/dmin  -(max - q)/w^2      d/dmax  -(q - min)/w^2
##
## and the upper tail S = 1 - F just flips every sign. On the log scale each is
## divided by the probability itself, which cancels to closed forms with no
## logarithm left in them:
##
##   log F:   d/dq  1/(q - min)   d/dmin  -(max - q)/(w (q - min))   d/dmax  -1/w
##   log S:   d/dq -1/(max - q)   d/dmin   1/w    d/dmax  (q - min)/(w (max - q))
##
## Off the interior the CDF is pinned at a constant 0 or 1 and every derivative
## is 0. The comparisons are >=/<=, matching the at_or_above/at_or_below guards
## in nv_punif(), so the endpoints belong to the constant branches; at those
## kinks the derivative does not exist and this is a convention, not a claim.
##
## At +-Inf the true derivative is 0 and nv_punif() agrees, but only because
## its interior branch is fed a clamped stand-in for q. Without that guard the
## untaken branch of nv_ifelse() evaluates to +-Inf and the reverse pass
## combines it as 0 * Inf = NaN. Sweeping +-Inf rather than gating it out is
## what keeps that guard honest.
punif_grad_ref <- function(q, p, f) {
  a <- p$min
  b <- p$max
  w <- b - a
  lower <- isTRUE(f$lower_tail)
  interior <- !is.nan(q) & q > a & q < b
  z <- if (!isTRUE(f$log_p)) {
    list(
      q = if (lower) 1 / w else -1 / w,
      min = (if (lower) -1 else 1) * (b - q) / w^2,
      max = (if (lower) -1 else 1) * (q - a) / w^2
    )
  } else if (lower) {
    list(q = 1 / (q - a), min = -(b - q) / (w * (q - a)), max = -1 / w)
  } else {
    list(q = -1 / (b - q), min = 1 / w, max = (q - a) / (w * (b - q)))
  }
  lapply(z, function(v) ifelse(is.nan(q), NaN, ifelse(interior, v, 0)))
}

## A stable reference for the log-scale CDF. base R's punif() forms the
## probability and then takes its log (log(1 - t) for the upper tail), so where
## the probability is within an ulp of 1 the log is lost: at q = 1e-100 on
## [0, 1], upper tail, it returns 0 where the answer is log1p(-1e-100). This
## takes the small tail directly on each side:
##
##   u = (q - min)/w,  v = (max - q)/w
##   log F = log(u)      if u <= 1/2,  else log1p(-v)
##   log S = log(v)      if v <= 1/2,  else log1p(-u)
##
## so near either endpoint the quantity that is small is the one computed, and
## the cancellation base R suffers never happens. Endpoints and beyond are the
## constants 0 / -Inf, with the >= / <= comparisons base R uses.
##
## This is the same branch structure as nv_punif() itself (R/api-distributions.R).
## It is evaluated in double, 29 bits beyond an f32 result, but for f64 cells
## it shares anvl's formula at anvl's precision -- which is why its bound is
## checked against high precision rather than trusted.
##
## Error bound, declared in double ulps at the result and to be validated: w,
## q - min and max - q each round by at most 1/2 ulp, so u and v carry at most
## ~1.5 ulp relative error; log(u) with u <= 1/2 has |log u| >= log 2 and turns
## that into <= ~2.2 ulp, log1p(-v) with v <= 1/2 into <= ~3 ulp, plus 1 ulp
## for log/log1p themselves. 8 leaves margin over that ~4.
punif_log_stable <- function(q, p, f) {
  a <- p$min
  b <- p$max
  w <- b - a
  lower <- isTRUE(f$lower_tail)
  out <- rep(NaN, length(q))
  lo <- !is.na(q) & q <= a
  hi <- !is.na(q) & q >= b
  out[lo] <- if (lower) -Inf else 0
  out[hi] <- if (lower) 0 else -Inf
  i <- which(!is.na(q) & !lo & !hi)
  if (length(i)) {
    u <- (q[i] - a) / w
    v <- (b - q[i]) / w
    small <- if (lower) u else v
    other <- if (lower) v else u
    out[i] <- ifelse(small <= 0.5, log(pmax(small, 0)), log1p(-other))
  }
  out
}

grad_punif <- anvl::jit(
  anvl::gradient(
    \(q, min, max, lower_tail = TRUE, log_p = FALSE) {
      sum(anvl::nv_punif(q, min, max, lower_tail = lower_tail, log_p = log_p))
    },
    wrt = c("q", "min", "max")
  ),
  static = c("lower_tail", "log_p")
)

sweep_spec(
  name = "nv_punif",
  family = "uniform",
  params = UNIF_INTERVALS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),

  ## punif() is defined on the whole line and never warns, so nothing off the
  ## interval is excused: a disagreement there is a real finding. The support
  ## is reported, and excuses nothing.
  domain = function(p, f) c(-Inf, Inf),
  support = function(p, f) c(p$min, p$max),
  ## With log_p, nv_punif switches from log(u) to log1p(-v) of the opposite
  ## tail where u = 1/2, i.e. at the interval's midpoint; u is computed at the
  ## cell's precision, so the switch can sit an ulp either side, which the
  ## midpoint's neighbours cover. See R/api-distributions.R, nv_punif.
  branch_points = function(p, f, dtype) {
    if (isTRUE(f$log_p)) c(log_log1p_switch = p$min + (p$max - p$min) / 2)
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_punif(
      anvl::nv_array(x, dtype = dtype),
      p$min,
      p$max,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    punif(x, min = p$min, max = p$max, lower.tail = f$lower_tail, log.p = f$log_p)
  },
  ref_stable = punif_log_stable,
  ref_stable_mpfr = function(x, p, f) {
    q <- mp_num(x)
    a <- mp_num(p$min)
    b <- mp_num(p$max)
    lower <- isTRUE(f$lower_tail)
    w <- p$max - p$min
    out <- x
    out[!is.nan(q) & q <= a] <- if (lower) -Inf else 0
    out[!is.nan(q) & q >= b] <- if (lower) 0 else -Inf
    ## the tail that is small, directly, even at 256 bits: 1 - 4e-78 is
    ## exactly 1 there, so log(1 - u) would lose the answer just as base R does
    i <- which(!is.nan(q) & q > a & q < b)
    if (length(i)) {
      u <- (x[i] - p$min) / w
      v <- (p$max - x[i]) / w
      small <- if (lower) u else v
      other <- if (lower) v else u
      half <- mp_num(small) <= 0.5
      r <- log1p(-other)
      if (any(half)) {
        r[half] <- log(small[half])
      }
      out[i] <- r
    }
    out
  },
  ref_stable_bound_ulp64 = 8,
  ref_stable_covers = function(f) isTRUE(f$log_p),
  ref_stable_note = "base R takes the log of the probability, so it loses the log where the probability rounds to 1 (e.g. 0 instead of log1p(-q) near an endpoint)",

  grad_wrt = c("q", "min", "max"),
  grad = function(x, dtype, p, f) {
    n <- length(x)
    d <- grad_punif(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$min, n), dtype = dtype),
      anvl::nv_array(rep(p$max, n), dtype = dtype),
      lower_tail = f$lower_tail,
      log_p = f$log_p
    )
    lapply(list(q = d$q, min = d$min, max = d$max), as.double)
  },
  ref_grad = punif_grad_ref,
  ref_grad_bound_ulp64 = 4,
  ## The interior derivatives exactly, and punif_grad_ref's convention (0)
  ## everywhere else.
  ref_grad_mpfr = function(x, p, f) {
    q <- mp_num(x)
    a <- p$min
    b <- p$max
    w <- b - a
    lower <- isTRUE(f$lower_tail)
    z <- if (!isTRUE(f$log_p)) {
      sg <- if (lower) -1 else 1
      list(q = -sg / w * mp_fill(x, 1), min = sg * (b - x) / (w * w), max = sg * (x - a) / (w * w))
    } else if (lower) {
      list(q = 1 / (x - a), min = -(b - x) / (w * (x - a)), max = -1 / w * mp_fill(x, 1))
    } else {
      list(q = -1 / (b - x), min = 1 / w * mp_fill(x, 1), max = (x - a) / (w * (b - x)))
    }
    interior <- !is.nan(q) & q > mp_num(a) & q < mp_num(b)
    lapply(z, function(v) {
      v[!interior] <- 0
      v[is.nan(q)] <- NaN
      v
    })
  },

  ## jax.scipy.stats.uniform offers only `cdf` -- no logcdf and no survival
  ## function -- so only the lower-tail, non-log variants have a twin.
  jax_covers = function(f, kind) isTRUE(f$lower_tail) && !isTRUE(f$log_p),
  jax_value = function(x, dtype, p, f) {
    jax_init()
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    js <- reticulate::import("jax.scipy.stats", convert = FALSE)
    np <- reticulate::import("numpy")
    as.double(np$asarray(js$uniform$cdf(
      jnp$asarray(x, dtype = jax_dtype(dtype)),
      p$min,
      p$max - p$min
    )))
  },
  jax_grad = function(x, dtype, p, f) {
    jax_init()
    reticulate::py_run_string(
      "
import jax
from jax.scipy.stats import uniform as _u
def _punif_grad(i):
    g = jax.grad(lambda q, mn, mx: _u.cdf(q, mn, mx - mn), argnums=i)
    return jax.jit(jax.vmap(g, in_axes=(0, None, None)))
"
    )
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    np <- reticulate::import("numpy")
    xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
    g <- function(i) as.double(np$asarray(reticulate::py$`_punif_grad`(i)(xx, p$min, p$max)))
    list(q = g(0L), min = g(1L), max = g(2L))
  }
)
