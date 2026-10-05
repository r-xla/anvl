## ---------------------------------------------------------------------------
## nv_dexp against base R dexp().
##
## The non-log density is the interesting variant: nv_dexp() computes
## rate * exp(-t), t = rate * x, as (rate * exp(-t/2)) * exp(-t/2) past
## t = 700 (f64) / 80 (f32), so that a large rate can rescale the exponential
## before it underflows. The sweep crosses that seam from both sides, and the
## `large` rate is what reaches the region where the split changes the answer.
##
## No stable reference is declared, although base R is the weaker side in that
## region: it forms exp(-x/scale) first, which is subnormal for t in (708.4,
## 745) and keeps only a few bits, then divides by scale. A dispute needs anvl
## within a few ulp of the truth, and anvl rounds t like base R does, costing
## up to t/2 ulp (~360 at t = 720) -- so no sample there could qualify, and
## the reference would only add work. The disagreement is reported as it is.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_exponential.R"), local = TRUE)

## With t = rate * x and f = rate exp(-t) the density on the support x >= 0,
##
##   d/dx     -rate^2 exp(-t)      log:  -rate
##   d/drate  (1 - t) exp(-t)      log:  (1 - t) / rate
##
## Off the support (x < 0) the density is the constant 0 (resp. -Inf) and every
## derivative is 0; at x = +Inf it is 0 too, and nv_dexp() resolves it to the
## constant directly. The support is closed -- x == 0 is in it -- matching the
## `x < 0` test in nv_dexp(); the density jumps there, so this is a convention,
## not a claim.
##
## t is carried as hi + lo (rate_times(), _exponential.R): the rounding of t
## costs exp(-t) about t ulp, and 1 - t cancels near t = 1, where 1 - hi is
## exact (Sterbenz) and lo then supplies the rest. Where t overflows (x near
## the top of the double range at the `large` rate) the log-scale d/drate is
## still finite: (1 - t)/rate = 1/rate - x, which does not cancel there. The
## density-scale forms go through exp_times() so that a large rate^2 or 1 - t
## is applied before exp(-t) underflows.
dexp_grad_ref <- function(x, p, f) {
  r <- p$rate
  tt <- rate_times(x, r)
  t <- tt$hi
  lo <- tt$lo
  one_minus_t <- (1 - t) - lo
  d <- if (isTRUE(f$log)) {
    huge <- !is.na(t) & abs(t) >= 1e290
    list(x = rep(-r, length(x)), rate = ifelse(huge, 1 / r - x, one_minus_t / r))
  } else {
    list(x = exp_times(t, -r * r, lo), rate = exp_times(t, one_minus_t, lo))
  }
  on <- !is.nan(x) & x >= 0 & x < Inf
  lapply(d, function(v) ifelse(is.nan(x), NaN, ifelse(on, v, 0)))
}

grad_dexp <- anvl::jit(
  anvl::gradient(
    \(x, rate, log = FALSE) sum(anvl::nv_dexp(x, rate, log = log)),
    wrt = c("x", "rate")
  ),
  static = "log"
)

sweep_spec(
  name = "nv_dexp",
  family = "exponential",
  params = EXP_PARAMS,
  flags = list(log = c(FALSE, TRUE)),

  ## Defined on the whole line. Below the support the density is the constant
  ## 0 (or -Inf on the log scale), so a disagreement there is a real finding:
  ## the support is reported, and excuses nothing.
  domain = function(p, f) c(-Inf, Inf),
  support = function(p, f) c(0, Inf),
  ## Without log, nv_dexp splits the exponential where -rate * x < -80 (f32) or
  ## -700 (f64); t is formed at the cell's precision, so the switch can sit an
  ## ulp either side, which the point's neighbours cover. See
  ## R/api-distributions.R, nv_dexp.
  branch_points = function(p, f, dtype) {
    if (!isTRUE(f$log)) c(exp_split = (if (dtype == "f32") 80 else 700) / p$rate)
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_dexp(anvl::nv_array(x, dtype = dtype), p$rate, log = f$log))
  },
  ref_value = function(x, p, f) dexp(x, rate = p$rate, log = f$log),

  grad_wrt = c("x", "rate"),
  ## The rate is passed as a full-length array so that each element carries
  ## its own d/drate; with a scalar rate the gradient would be one summed
  ## number and there would be nothing to sweep.
  grad = function(x, dtype, p, f) {
    d <- grad_dexp(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$rate, length(x)), dtype = dtype),
      log = f$log
    )
    lapply(list(x = d$x, rate = d$rate), as.double)
  },
  ref_grad = dexp_grad_ref,
  ## t as hi + lo leaves exp(-t) its own rounding, at most two roundings more
  ## through the halves past t = 700, and one each for the multiplier, its
  ## exp(-lo) correction and the product: ~4. 8 leaves margin.
  ref_grad_bound_ulp64 = 8,
  ref_grad_mpfr = function(x, p, f) {
    xn <- mp_num(x)
    t <- p$rate * x
    d <- if (isTRUE(f$log)) {
      list(x = -p$rate * mp_fill(x, 1), rate = (1 - t) / p$rate)
    } else {
      e <- exp(-t)
      list(x = -(p$rate * p$rate) * e, rate = (1 - t) * e)
    }
    on <- !is.nan(xn) & xn >= 0 & xn < Inf
    lapply(d, function(v) {
      v[!on] <- 0
      v[is.nan(xn)] <- NaN
      v
    })
  },

  ## jax.scipy.stats.expon covers both pdf and logpdf, so every variant has a
  ## twin.
  jax_value = function(x, dtype, p, f) {
    jax_expon_value(if (isTRUE(f$log)) "logpdf" else "pdf", x, dtype, p$rate)
  },
  jax_grad = function(x, dtype, p, f) {
    jax_expon_grad(if (isTRUE(f$log)) "logpdf" else "pdf", x, dtype, p$rate, c("x", "rate"))
  }
)
