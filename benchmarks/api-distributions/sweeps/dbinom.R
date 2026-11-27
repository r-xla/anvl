## ---------------------------------------------------------------------------
## nv_dbinom against base R dbinom(), swept in prob (see _binomial.R for why).
##
## nv_dbinom() evaluates Loader's saddle point expansion, as base R does, but
## forms x - n p from the exact product and log(1 - x / n) as
## log(n - x) - log(n) above x = n / 2. Base R rounds n p, which costs it up
## to ~1e-9 relative at size 1e13; JAX's logpmf differences three lgamma
## values of size ~n log n and loses ~1e-2 there. The stable reference below
## is what lets the sweep tell base R's error from anvl's.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_binomial.R"), local = TRUE)

## With f the density and d = x - n p,
##
##   d/dp f      f d / (p (1 - p))         log:  d / (p (1 - p))
##
## and at the ends of [0, 1], where that form is 0 / 0, its limits:
##
##   p = 0:  f' = n [x = 1] - n [x = 0]    log:  -n at x = 0, +Inf otherwise
##   p = 1:  f' = n [x = n] - n [x = n-1]  log:   n at x = n, -Inf otherwise
##
## (the density is the polynomial C(n, x) p^x (1 - p)^(n - x), so at p = 0 only
## its linear term survives, at x = 1). On the log scale anvl returns 0 where
## the log density is -Inf, a documented convention; the reference keeps the
## mathematical limit, so those disagreements are recorded at the domain
## endpoints as boundary behaviour. Outside [0, 1] the density is NaN and
## every derivative is 0.
##
## d is exact (binom_np()), and the density-scale form is taken as
## sign(d) exp(log f + log|d| - log p - log1p(-p)), so that a density that
## has underflowed does not take a representable derivative with it.
dbinom_grad_ref <- function(pr, p, f) {
  x <- p$x
  n <- p$size
  out <- rep(0, length(pr))
  out[is.na(pr)] <- NaN
  log <- isTRUE(f$log)
  i <- which(!is.na(pr) & pr > 0 & pr < 1)
  if (length(i)) {
    q <- pr[i]
    np <- binom_np(n, q)
    d <- (x - np$hi) - np$lo
    out[i] <- if (log) {
      d / (q * (1 - q))
    } else {
      sign(d) * exp(binom_log_density(q, x, n) + log(abs(d)) - log(q) - log1p(-q))
    }
  }
  at0 <- which(!is.na(pr) & pr == 0)
  out[at0] <- if (log) (if (x == 0) -n else Inf) else n * ((x == 1) - (x == 0))
  at1 <- which(!is.na(pr) & pr == 1)
  out[at1] <- if (log) (if (x == n) n else -Inf) else n * ((x == n) - (x == n - 1))
  list(prob = out)
}

grad_dbinom <- anvl::jit(
  anvl::gradient(
    \(x, size, prob, log = FALSE) sum(anvl::nv_dbinom(x, size, prob, log = log)),
    wrt = "prob"
  ),
  static = "log"
)

sweep_spec(
  name = "nv_dbinom",
  family = "binomial",
  params = BINOM_D_PARAMS,
  flags = list(log = c(FALSE, TRUE)),

  ## The swept argument is prob. Outside [0, 1] base R returns NaN with a
  ## warning; a value disagreement there is still a failure.
  domain = function(p, f) c(0, 1),
  support = function(p, f) c(0, 1),
  ## Loader's deviance terms bd0(x, n p) and bd0(n - x, n (1 - p)) switch from
  ## x log(x / m) + m - x to their series where |x - m| < 0.1 (x + m), i.e. at
  ## n p = 9 x / 11 and 11 x / 9, and the same for n - x. Only interior x has
  ## them. x - n p is formed at the cell's precision, so a switch can sit an
  ## ulp either side, which the points' neighbours cover. See
  ## R/api-distributions.R, binom_bd0 and binom_log_density.
  branch_points = function(p, f, dtype) {
    x <- p$x
    n <- p$size
    if (x > 0 && x < n) {
      bp <- c(
        bd0_x_below = 9 * x / (11 * n),
        bd0_x_above = 11 * x / (9 * n),
        bd0_nx_below = 1 - 11 * (n - x) / (9 * n),
        bd0_nx_above = 1 - 9 * (n - x) / (11 * n)
      )
      bp[bp > 0 & bp < 1]
    }
  },
  ## x is the operand that sets the precision, so it is passed as an array of
  ## the cell's dtype; prob is the swept array, size an R value.
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_dbinom(
      anvl::nv_scalar(p$x, dtype = dtype),
      p$size,
      anvl::nv_array(x, dtype = dtype),
      log = f$log
    ))
  },
  ref_value = function(x, p, f) suppressWarnings(dbinom(p$x, p$size, x, log = f$log)),
  ref_stable = binom_density_stable,
  ref_stable_mpfr = function(x, p, f) {
    pr <- mp_num(x)
    xn <- mp_num(p$x)
    nn <- mp_num(p$size)
    ld <- mp_binom_log_density(x, p$x, p$size)
    out <- if (isTRUE(f$log)) ld else exp(ld)
    ## the ends of [0, 1], where 0 log 0 is 0 and the logs are -Inf
    out[!is.nan(pr) & pr == 0] <- if (xn == 0) (if (isTRUE(f$log)) 0 else 1) else (if (isTRUE(f$log)) -Inf else 0)
    out[!is.nan(pr) & pr == 1] <- if (xn == nn) (if (isTRUE(f$log)) 0 else 1) else (if (isTRUE(f$log)) -Inf else 0)
    out[is.nan(pr) | pr < 0 | pr > 1] <- NaN
    out
  },
  ## The log density is a sum of terms each up to about |log f| + 20 in size
  ## (the deviance terms, Stirling corrections and log(2 pi x (n - x) / n)),
  ## and each subtraction rounds at their scale: an absolute error of about
  ## 5 ulp(|log f| + 20), up to ~5 * 256 f64 ulps of the density near the
  ## bottom of the double range, where |log f| ~ 708. 4096 leaves margin over
  ## that; base R's error at size 1e13 is ~1e-9, some 4e6 ulps.
  ref_stable_bound_ulp64 = 4096,
  ref_stable_note = "base R forms x - n p from a rounded n p, which costs Loader's deviance terms about d ulp(n p) / (n p): ~1e-9 relative at size 1e13; and log1p(-x / n) loses the digits of n - x near x = n",

  grad_wrt = "prob",
  grad = function(x, dtype, p, f) {
    d <- grad_dbinom(
      anvl::nv_scalar(p$x, dtype = dtype),
      anvl::nv_scalar(p$size, dtype = dtype),
      anvl::nv_array(x, dtype = dtype),
      log = f$log
    )
    list(prob = as.double(d$prob))
  },
  ref_grad = dbinom_grad_ref,
  ## As the stable density: the density-scale derivative carries its error,
  ## plus ~3 roundings for log|d| - log p - log1p(-p); the log-scale one is d
  ## (two roundings) over p (1 - p): ~4. 4096 covers the former.
  ref_grad_bound_ulp64 = 4096,
  ref_grad_mpfr = function(x, p, f) {
    pr <- mp_num(x)
    xn <- mp_num(p$x)
    nn <- mp_num(p$size)
    d <- p$x - p$size * x
    out <- if (isTRUE(f$log)) {
      d / (x * (1 - x))
    } else {
      exp(mp_binom_log_density(x, p$x, p$size)) * d / (x * (1 - x))
    }
    log <- isTRUE(f$log)
    out[!is.nan(pr) & pr == 0] <- if (log) (if (xn == 0) -nn else Inf) else nn * ((xn == 1) - (xn == 0))
    out[!is.nan(pr) & pr == 1] <- if (log) (if (xn == nn) nn else -Inf) else nn * ((xn == nn) - (xn == nn - 1))
    out[!is.nan(pr) & (pr < 0 | pr > 1)] <- 0
    out[is.nan(pr)] <- NaN
    list(prob = out)
  },

  ## jax.scipy.stats.binom has pmf and logpmf, so every variant has a twin.
  jax_value = function(x, dtype, p, f) jax_dbinom_value(x, dtype, p, isTRUE(f$log)),
  jax_grad = function(x, dtype, p, f) jax_dbinom_grad(x, dtype, p, isTRUE(f$log))
)
