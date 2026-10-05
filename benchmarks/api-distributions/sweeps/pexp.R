## ---------------------------------------------------------------------------
## nv_pexp against base R pexp().
##
## The lower-tail log_p variant is the interesting one. nv_pexp() takes
## log(1 - exp(-t)), t = rate * q, by Maechler's log1mexp split at t = log 2,
## and where t underflows to zero it switches to the log of the product
## rescaled into range. The sweep crosses the log1mexp seam at every rate; the
## `small` rate is what reaches the underflow branch, in both precisions.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_exponential.R"), local = TRUE)

## With t = rate * q, on the interior 0 < q < Inf the lower-tail CDF is
## F = 1 - exp(-t) and the upper tail S = exp(-t), so
##
##   F:      d/dq  rate exp(-t)       d/drate  q exp(-t)
##   S:      the same, negated
##   log F:  d/dq  rate / expm1(t)    d/drate  q / expm1(t)
##   log S:  d/dq  -rate              d/drate  -q
##
## At and below q = 0 the CDF is pinned at a constant and every derivative is
## 0; the comparison is <=, matching the at_or_below guard in nv_pexp(), so
## q = 0 belongs to the constant branch -- a kink, where the derivative does
## not exist and this is a convention, not a claim. At q = +Inf nv_pexp()
## resolves the constant directly, and the true derivative is 0 there too.
##
## t is carried as hi + lo (rate_times(), _exponential.R), and the
## probability-scale forms go through exp_times() so that nothing underflows
## before the answer does: q exp(-t) is still representable well past
## exp(-t)'s underflow when q is large. On the log scale:
##
##   t < 2^-53     m / expm1(t) = (m / t)(1 - t/2 + ...) and the correction is
##                 below double precision: exactly 1/q and 1/rate, which also
##                 holds where t itself has underflowed;
##   t <= 700      m / (expm1(hi) + exp(hi) lo), lo folded in to first order;
##   t > 700       expm1(t) = exp(t) to double precision: exp_times(t, m).
pexp_grad_ref <- function(q, p, f) {
  r <- p$rate
  tt <- rate_times(q, r)
  t <- tt$hi
  lo <- tt$lo
  lower <- isTRUE(f$lower_tail)
  s <- if (lower) 1 else -1
  d <- if (!isTRUE(f$log_p)) {
    list(q = s * exp_times(t, r, lo), rate = s * exp_times(t, q, lo))
  } else if (!lower) {
    list(q = rep(-r, length(q)), rate = -q)
  } else {
    over_expm1 <- function(m) {
      out <- m / (expm1(t) + exp(t) * lo)
      far <- which(!is.na(t) & t > 700)
      out[far] <- exp_times(t[far], m[far], lo[far])
      out
    }
    dq <- over_expm1(rep(r, length(q)))
    dr <- over_expm1(q)
    tiny <- which(!is.na(t) & t < 2^-53)
    dq[tiny] <- 1 / q[tiny]
    dr[tiny] <- 1 / r
    list(q = dq, rate = dr)
  }
  interior <- !is.nan(q) & q > 0 & q < Inf
  lapply(d, function(v) ifelse(is.nan(q), NaN, ifelse(interior, v, 0)))
}

## A stable reference for the lower-tail log CDF. base R's pexp() forms
## x / scale, scale = 1/rate, and takes log1mexp of it directly. Where that
## quotient is subnormal it keeps only a few bits, and the log of it is off by
## far more than an ulp; below the smallest subnormal it is 0 and the log is
## -Inf. With the `small` rate that happens for every q below ~7e-299, where
## the answer is still an ordinary log(rate) + log(q). log_cdf_lower()
## (_exponential.R) takes that form there, and log1mexp of the exact product
## elsewhere. Endpoints and beyond are the constants -Inf / 0, with the <=
## comparison base R uses.
##
## Error bound, declared in double ulps at the result and to be validated: t is
## exact as hi + lo, so log(-expm1(-t)) and log1p(-exp(-t)) carry their own
## rounding (<= 1 ulp each) plus ~1 ulp for folding lo in and for the product
## inside log1p; log(rate) + log(q) is two roundings and a sum that does not
## cancel. 8 leaves margin over that ~3.
pexp_log_lower_stable <- function(q, p, f) {
  out <- rep(NaN, length(q))
  out[!is.na(q) & q <= 0] <- -Inf
  out[!is.na(q) & q == Inf] <- 0
  i <- which(!is.na(q) & q > 0 & q < Inf)
  if (length(i)) {
    out[i] <- log_cdf_lower(q[i], p$rate)
  }
  out
}

grad_pexp <- anvl::jit(
  anvl::gradient(
    \(q, rate, lower_tail = TRUE, log_p = FALSE) {
      sum(anvl::nv_pexp(q, rate, lower_tail = lower_tail, log_p = log_p))
    },
    wrt = c("q", "rate")
  ),
  static = c("lower_tail", "log_p")
)

## The jax.scipy.stats.expon function matching a flag combination.
pexp_jax_name <- function(f) {
  if (isTRUE(f$lower_tail)) {
    if (isTRUE(f$log_p)) "logcdf" else "cdf"
  } else {
    if (isTRUE(f$log_p)) "logsf" else "sf"
  }
}

sweep_spec(
  name = "nv_pexp",
  family = "exponential",
  params = EXP_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),

  ## pexp() is defined on the whole line and never warns, so nothing below the
  ## support is excused: a disagreement there is a real finding. The support
  ## is reported, and excuses nothing.
  domain = function(p, f) c(-Inf, Inf),
  support = function(p, f) c(0, Inf),
  ## With log_p in the lower tail, nv_pexp switches from log(-expm1(-t)) to
  ## log1p(-exp(-t)) at t = rate * q = log 2, and to the rescaled log of the
  ## product where t underflows, i.e. below the smallest normal. t is formed
  ## at the cell's precision, so each switch can sit an ulp either side, which
  ## the points' neighbours cover. See R/api-distributions.R, nv_pexp and
  ## log1mexp.
  branch_points = function(p, f, dtype) {
    if (isTRUE(f$lower_tail) && isTRUE(f$log_p)) {
      c(
        log1mexp_switch = log(2) / p$rate,
        product_underflow = (if (dtype == "f32") 2^-126 else 2^-1022) / p$rate
      )
    }
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_pexp(
      anvl::nv_array(x, dtype = dtype),
      p$rate,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    pexp(x, rate = p$rate, lower.tail = f$lower_tail, log.p = f$log_p)
  },
  ref_stable = pexp_log_lower_stable,
  ref_stable_mpfr = function(x, p, f) {
    q <- mp_num(x)
    out <- x
    out[!is.nan(q) & q <= 0] <- -Inf
    out[!is.nan(q) & q == Inf] <- 0
    i <- which(!is.nan(q) & q > 0 & q < Inf)
    if (length(i)) {
      out[i] <- mp_log1mexp(-(p$rate * x[i]))
    }
    out
  },
  ref_stable_bound_ulp64 = 8,
  ref_stable_covers = function(f) isTRUE(f$lower_tail) && isTRUE(f$log_p),
  ref_stable_note = "base R takes log1mexp of x / scale, which is subnormal or zero where rate * q underflows, so it loses the log there (e.g. -Inf instead of log(rate) + log(q))",

  grad_wrt = c("q", "rate"),
  grad = function(x, dtype, p, f) {
    d <- grad_pexp(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$rate, length(x)), dtype = dtype),
      lower_tail = f$lower_tail,
      log_p = f$log_p
    )
    lapply(list(q = d$q, rate = d$rate), as.double)
  },
  ref_grad = pexp_grad_ref,
  ## As for nv_dexp's references: t exact, exp(-t) and expm1(t) their own
  ## rounding, the halves past t = 700, and the multiplier, its correction and
  ## the product or quotient: ~4. 8 leaves margin.
  ref_grad_bound_ulp64 = 8,
  ref_grad_mpfr = function(x, p, f) {
    q <- mp_num(x)
    t <- p$rate * x
    lower <- isTRUE(f$lower_tail)
    s <- if (lower) 1 else -1
    d <- if (!isTRUE(f$log_p)) {
      e <- exp(-t)
      list(q = s * p$rate * e, rate = s * x * e)
    } else if (!lower) {
      list(q = -p$rate * mp_fill(x, 1), rate = -x)
    } else {
      em1 <- expm1(t)
      list(q = p$rate / em1, rate = x / em1)
    }
    interior <- !is.nan(q) & q > 0 & q < Inf
    lapply(d, function(v) {
      v[!interior] <- 0
      v[is.nan(q)] <- NaN
      v
    })
  },

  ## jax.scipy.stats.expon has cdf, logcdf, sf and logsf, so all four
  ## variants have a twin.
  jax_value = function(x, dtype, p, f) {
    jax_expon_value(pexp_jax_name(f), x, dtype, p$rate)
  },
  jax_grad = function(x, dtype, p, f) {
    jax_expon_grad(pexp_jax_name(f), x, dtype, p$rate, c("q", "rate"))
  }
)
