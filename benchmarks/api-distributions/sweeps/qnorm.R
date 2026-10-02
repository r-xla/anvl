## ---------------------------------------------------------------------------
## nv_qnorm against base R qnorm().
##
## nv_qnorm tests its regime predicate on whichever quantity the caller
## supplied -- p when a probability was given, log p when a log-probability
## was -- because the flag is static and an R-level `if` can hand each branch
## its exact input. The two log_p variants therefore traverse genuinely
## different code, and the seam-straddling region near p = 1/2 (where the
## centre branch gives way, and where p - 1/2 must be formed through expm1 on
## the log scale) is the part most worth sweeping densely.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_normal.R"), local = TRUE)

## x = mean + sd * z with z the standard quantile, so
##
##   d/dp  sd * dz/dp      d/dmean  1      d/dsd  z
##
## and dz/dp is the reciprocal of the density at z, negated for the upper tail
## and multiplied by p itself on the log scale. On the probability scale the
## reciprocal goes through phi_recip() (_normal.R): a plain division returns
## Inf once the density underflows although the reciprocal is still finite,
## and exp(-log phi) turns the rounding of a log of ~-450 into ~5e-14 relative
## error.
qnorm_grad_ref <- function(pr, p, f) {
  lower <- isTRUE(f$lower_tail)
  s <- if (lower) 1 else -1
  ## the quantile as hi + lo: base R's qnorm() is a few ulp out, which the
  ## derivative multiplies by |z| (see std_q() in _normal.R)
  q <- std_q(pr, lower, isTRUE(f$log_p))
  z <- q$hi
  zl <- q$lo
  ## dz/dp is 1/phi(z); on the log scale it is p/phi(z), the reciprocal of
  ## the inverse Mills ratio, never exp(log p - log phi), which cancels in the
  ## far tail. m(z + lo) = m(z) (1 - (z + m) lo) from m' = -m (z + m).
  d_p <- if (isTRUE(f$log_p)) {
    zs <- s * z
    ls <- s * zl
    m0 <- inv_mills(zs)
    p$sd * s / ifelse(ls == 0, m0, m0 * (1 - (zs + m0) * ls))
  } else {
    phi_recip(z, rep(s * p$sd, length(z)), zl)
  }
  in_range <- if (isTRUE(f$log_p)) !is.nan(pr) & pr <= 0 else !is.nan(pr) & pr >= 0 & pr <= 1
  z0 <- list(p = d_p, mean = rep(1, length(pr)), sd = z + zl)
  lapply(z0, function(v) ifelse(is.nan(pr), NaN, ifelse(in_range, v, 0)))
}

grad_qnorm <- anvl::jit(
  anvl::gradient(
    \(p, mean, sd, lower_tail = TRUE, log_p = FALSE) {
      sum(anvl::nv_qnorm(p, mean, sd, lower_tail = lower_tail, log_p = log_p))
    },
    wrt = c("p", "mean", "sd")
  ),
  static = c("lower_tail", "log_p")
)

sweep_spec(
  name = "nv_qnorm",
  family = "normal",
  params = NORM_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),
  domain = function(p, f) if (isTRUE(f$log_p)) c(-Inf, 0) else c(0, 1),
  ## nv_qnorm switches between central and tail approximations at p = e^-2 and
  ## 1 - e^-2, and between near and far tail at z = 8, i.e. t = e^-32 on either
  ## side. Tested on p, or on log p when log_p = TRUE. See R/api-distributions.R.
  branch_points = function(p, f, dtype) {
    if (isTRUE(f$log_p)) {
      c(lower_central = -2, upper_central = log1p(-exp(-2)), lower_far = -32, upper_far = log1p(-exp(-32)))
    } else {
      c(lower_central = exp(-2), upper_central = 1 - exp(-2), lower_far = exp(-32), upper_far = 1 - exp(-32))
    }
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_qnorm(
      anvl::nv_array(x, dtype = dtype),
      p$mean,
      p$sd,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    suppressWarnings(qnorm(x, mean = p$mean, sd = p$sd, lower.tail = f$lower_tail, log.p = f$log_p))
  },

  grad_wrt = c("p", "mean", "sd"),
  grad = function(x, dtype, p, f) {
    n <- length(x)
    d <- grad_qnorm(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$mean, n), dtype = dtype),
      anvl::nv_array(rep(p$sd, n), dtype = dtype),
      lower_tail = f$lower_tail,
      log_p = f$log_p
    )
    lapply(list(p = d$p, mean = d$mean, sd = d$sd), as.double)
  },
  ref_grad = qnorm_grad_ref,
  ref_grad_bound_ulp64 = 16,
  ref_grad_mpfr = function(x, p, f) {
    pr <- mp_num(x) # exact: x came from a double
    lower <- isTRUE(f$lower_tail)
    s <- if (lower) 1 else -1
    z <- mp_qnorm(pr, lower, isTRUE(f$log_p), mp_prec(x))
    d_p <- if (isTRUE(f$log_p)) p$sd * s / mp_imills(s * z) else p$sd * s / mp_phi(z)
    in_range <- if (isTRUE(f$log_p)) !is.nan(pr) & pr <= 0 else !is.nan(pr) & pr >= 0 & pr <= 1
    out <- list(p = d_p, mean = mp_fill(x, 1), sd = z)
    lapply(out, function(v) {
      v[!in_range & !is.nan(pr)] <- 0
      v[is.nan(pr)] <- NaN
      v
    })
  },

  ## jax.scipy.stats.norm has ppf (lower tail) and isf (upper tail), both on
  ## the probability scale only -- there is no log_p quantile anywhere in JAX,
  ## which is precisely the gap nv_qnorm's log_p variants fill.
  jax_covers = function(f, kind) !isTRUE(f$log_p),
  jax_value = function(x, dtype, p, f) {
    jax_init()
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    js <- reticulate::import("jax.scipy.stats", convert = FALSE)
    np <- reticulate::import("numpy")
    fn <- if (isTRUE(f$lower_tail)) js$norm$ppf else js$norm$isf
    as.double(np$asarray(fn(jnp$asarray(x, dtype = jax_dtype(dtype)), p$mean, p$sd)))
  },
  jax_grad = function(x, dtype, p, f) {
    jax_init()
    reticulate::py_run_string(
      "
import jax
from jax.scipy.stats import norm as _n
def _qnorm_grad(lower, i):
    f = _n.ppf if lower else _n.isf
    return jax.jit(jax.vmap(jax.grad(f, argnums=i), in_axes=(0, None, None)))
"
    )
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    np <- reticulate::import("numpy")
    xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
    g <- function(i) {
      as.double(np$asarray(reticulate::py$`_qnorm_grad`(f$lower_tail, i)(xx, p$mean, p$sd)))
    }
    list(p = g(0L), mean = g(1L), sd = g(2L))
  }
)
