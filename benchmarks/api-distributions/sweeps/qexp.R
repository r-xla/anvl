## ---------------------------------------------------------------------------
## nv_qexp against base R qexp().
##
## The lower-tail log_p variant is the interesting one: nv_qexp() takes
## -log(1 - exp(p)) / rate by Maechler's log1mexp split at p = -log 2, and
## below p = -700 (f64) / -80 (f32) it splits the exponential instead, as
## (exp(p/2) / rate) * exp(p/2), so that a small rate can rescale exp(p) before
## it underflows. The sweep crosses both seams at every rate; the `small` rate
## is what reaches the region where the split changes the answer.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_exponential.R"), local = TRUE)

## x = -L / rate, with L the log of the upper-tail probability: log(1 - p),
## log p, log(1 - exp(p)) or p itself for the four variants. Hence
##
##   d/drate  L / rate^2 = -x / rate
##   d/dp     -L'(p) / rate:
##              lower       1 / (rate (1 - p))
##              upper      -1 / (rate p)
##              lower, log  1 / (rate expm1(-p))
##              upper, log -1 / rate
##
## Outside the valid range of p the result is NaN and every derivative is 0.
## At the ends of the range the derivative is taken one-sided, as for qunif:
## the formula holds up to the endpoint and gives its limiting value there
## (e.g. 1/rate at p = 0, lower tail). nv_qexp() resolves the quantile at
## probability zero to the constant 0, so its own d/dp there is 0; the sweep
## reports that difference at the exact point rather than this reference
## adopting it.
##
## In the lower tail on the log scale, below p = -700 exp(p) is near or below
## the smallest normal while exp(p) / rate may be ordinary for a small rate:
## those derivatives go through exp_over() (_exponential.R), and 1 - exp(p)
## is 1 to double precision there. Every d/drate divides by rate^2 in one
## step: L can be subnormal (log1p(-p) at a subnormal p), and dividing it by
## a small rate twice would round the intermediate to a subnormal first.
## expm1(-p) is taken as expm1(|p|) so that p = +0 gives +0 and the limiting
## d/dp is +Inf, not -Inf; likewise -1 / (rate p) as -1 / (rate |p|), so that
## p = -0 gives -Inf, the limit as p -> 0 from above, not +Inf.
qexp_grad_ref <- function(pr, p, f) {
  r <- p$rate
  lower <- isTRUE(f$lower_tail)
  log_p <- isTRUE(f$log_p)
  in_range <- !is.nan(pr) & (if (log_p) pr <= 0 else pr >= 0 & pr <= 1)
  ## out of range the derivative is 0 by convention; an interior stand-in keeps
  ## the transcendentals below from warning about values that are discarded
  q <- ifelse(in_range, pr, if (log_p) -1 else 0.5)
  d <- if (log_p) {
    if (lower) {
      far <- q < -700
      list(
        p = ifelse(far, exp_over(q, r), (1 / r) / expm1(abs(q))),
        rate = ifelse(far, -exp_over(q, r * r), log1mexp_ref(q) / (r * r))
      )
    } else {
      list(p = rep(-1 / r, length(q)), rate = q / (r * r))
    }
  } else if (lower) {
    list(p = 1 / (r * (1 - q)), rate = log1p(-q) / (r * r))
  } else {
    list(p = -1 / (r * abs(q)), rate = log(q) / (r * r))
  }
  lapply(d, function(v) ifelse(is.nan(pr), NaN, ifelse(in_range, v, 0)))
}

## A stable reference for the lower-tail log quantile. base R's qexp() forms
## -scale * log1mexp(p), scale = 1/rate. Below p = -708, log1mexp(p) =
## -exp(p) is subnormal and keeps only a few bits; below p = -745 it is 0 and
## so is the quantile. With the `small` rate the quantile there is still an
## ordinary exp(p) / rate, which exp_over() (_exponential.R) forms without the
## underflow; above p = -700 it is -log1mexp(p) / rate by the usual split.
## Endpoints and beyond follow base R: 0 at p = -Inf, Inf at p = 0, NaN for
## p > 0.
##
## Error bound, declared in double ulps at the result and to be validated:
## log1mexp's two forms carry their own rounding (<= 1 ulp) and the division
## another 1/2; exp_over() is two exponentials, a quotient and a product, each
## rounded once: ~2. 8 leaves margin.
qexp_log_lower_stable <- function(pr, p, f) {
  out <- rep(NaN, length(pr))
  out[!is.na(pr) & pr == -Inf] <- 0
  out[!is.na(pr) & pr == 0] <- Inf
  i <- which(!is.na(pr) & pr > -Inf & pr < 0)
  if (length(i)) {
    far <- pr[i] < -700
    out[i] <- ifelse(far, exp_over(pr[i], p$rate), -log1mexp_ref(pr[i]) / p$rate)
  }
  out
}

grad_qexp <- anvl::jit(
  anvl::gradient(
    \(p, rate, lower_tail = TRUE, log_p = FALSE) {
      sum(anvl::nv_qexp(p, rate, lower_tail = lower_tail, log_p = log_p))
    },
    wrt = c("p", "rate")
  ),
  static = c("lower_tail", "log_p")
)

sweep_spec(
  name = "nv_qexp",
  family = "exponential",
  params = EXP_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),

  ## A quantile function is only defined on its probability scale; base R
  ## returns NaN with a warning off it. A value disagreement there is still a
  ## failure (NaN specified, something else returned); only a gradient
  ## convention where both values are NaN is set aside.
  domain = function(p, f) if (isTRUE(f$log_p)) c(-Inf, 0) else c(0, 1),
  ## With log_p in the lower tail, nv_qexp switches from log(-expm1(p)) to
  ## log1p(-exp(p)) at p = -log 2, and to the split exponential below p = -80
  ## (f32) or -700 (f64). Tested on p itself. See R/api-distributions.R,
  ## nv_qexp and log1mexp.
  branch_points = function(p, f, dtype) {
    if (isTRUE(f$lower_tail) && isTRUE(f$log_p)) {
      c(log1mexp_switch = -log(2), exp_split = if (dtype == "f32") -80 else -700)
    }
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_qexp(
      anvl::nv_array(x, dtype = dtype),
      p$rate,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    suppressWarnings(qexp(x, rate = p$rate, lower.tail = f$lower_tail, log.p = f$log_p))
  },
  ref_stable = qexp_log_lower_stable,
  ref_stable_mpfr = function(x, p, f) {
    pr <- mp_num(x)
    out <- -mp_log1mexp(x) / p$rate
    out[!is.nan(pr) & pr == -Inf] <- 0
    out[!is.nan(pr) & pr == 0] <- Inf
    out[!is.nan(pr) & pr > 0] <- NaN
    out
  },
  ref_stable_bound_ulp64 = 8,
  ref_stable_covers = function(f) isTRUE(f$lower_tail) && isTRUE(f$log_p),
  ref_stable_note = "base R forms -scale * log1mexp(p), and log1mexp(p) = -exp(p) is subnormal or zero below p = -708, so the quantile is lost there (e.g. 0 instead of exp(p) / rate)",

  grad_wrt = c("p", "rate"),
  grad = function(x, dtype, p, f) {
    d <- grad_qexp(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$rate, length(x)), dtype = dtype),
      lower_tail = f$lower_tail,
      log_p = f$log_p
    )
    lapply(list(p = d$p, rate = d$rate), as.double)
  },
  ref_grad = qexp_grad_ref,
  ## Each derivative is one transcendental (log1p, log, expm1 or the log1mexp
  ## split) and up to three roundings for the products and quotients by rate,
  ## or exp_over()'s ~2: ~3. 8 leaves margin.
  ref_grad_bound_ulp64 = 8,
  ref_grad_mpfr = function(x, p, f) {
    pr <- mp_num(x)
    r <- p$rate
    lower <- isTRUE(f$lower_tail)
    if (isTRUE(f$log_p)) {
      in_range <- !is.nan(pr) & pr <= 0
      d <- if (lower) {
        list(p = 1 / (r * expm1(abs(x))), rate = mp_log1mexp(x) / (r * r))
      } else {
        list(p = -1 / r * mp_fill(x, 1), rate = x / (r * r))
      }
    } else {
      in_range <- !is.nan(pr) & pr >= 0 & pr <= 1
      d <- if (lower) {
        list(p = 1 / (r * (1 - x)), rate = log1p(-x) / (r * r))
      } else {
        list(p = -1 / (r * abs(x)), rate = log(x) / (r * r))
      }
    }
    lapply(d, function(v) {
      v[!in_range] <- 0
      v[is.nan(pr)] <- NaN
      v
    })
  },

  ## jax.scipy.stats.expon has ppf on the probability scale only -- no isf and
  ## no log_p quantile -- so only the lower-tail, non-log variant has a twin.
  jax_covers = function(f, kind) isTRUE(f$lower_tail) && !isTRUE(f$log_p),
  jax_value = function(x, dtype, p, f) {
    jax_expon_value("ppf", x, dtype, p$rate)
  },
  jax_grad = function(x, dtype, p, f) {
    jax_expon_grad("ppf", x, dtype, p$rate, c("p", "rate"))
  }
)
