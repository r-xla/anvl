## ---------------------------------------------------------------------------
## nv_dnorm against base R dnorm().
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_normal.R"), local = TRUE)

## With z = (x - mean)/sd and phi the density itself,
##
##   d/dx    -z/sd * phi        log:  -z/sd
##   d/dmean  z/sd * phi        log:   z/sd
##   d/dsd   (z^2 - 1)/sd * phi log:  (z^2 - 1)/sd
##
## Evaluated in an order that never under- or overflows before the answer does:
##
##   - (z^2 - 1)/sd as (z - 1) * ((z + 1)/sd): z^2 overflows near |z| = 1.3e154
##     although (z^2 - 1)/sd is finite for sd > 1 (4.03e307 at x = 1e155,
##     mean = -pi, sd = 2 pi), and near |z| = 1 the product has no cancellation,
##     z - 1 being exact there.
##   - the density-scale forms through phi_times() (_normal.R) rather than as a
##     bracket times dnorm(): phi underflows at |z| ~ 37.5 while the product is
##     still representable (1.7e-321 at x = 38.6 for d/dsd), and for huge |z|
##     the bracket overflows while phi is 0 and the answer is 0, not Inf * 0.
##
## The density is phi(z)/sd, so each derivative is phi_times(z, bracket / sd).
##
## z itself is carried as hi + lo (std_z(), _normal.R), so the reference does
## not inherit the ~z^2 ulp that rounding z to a double costs phi, nor the
## cancellation of z + 1 near z = -1 on the log scale.
dnorm_grad_ref <- function(x, p, f) {
  zz <- std_z(x, p$mean, p$sd)
  z <- zz$hi
  zl <- zz$lo
  core <- list(
    x = -(z + zl) / p$sd,
    mean = (z + zl) / p$sd,
    sd = ((z - 1) + zl) * (((z + 1) + zl) / p$sd)
  )
  if (isTRUE(f$log)) core else lapply(core, function(v) phi_times(z, v / p$sd, zl))
}

grad_dnorm <- anvl::jit(
  anvl::gradient(
    \(x, mean, sd, log = FALSE) sum(anvl::nv_dnorm(x, mean, sd, log = log)),
    wrt = c("x", "mean", "sd")
  ),
  static = "log"
)

sweep_spec(
  name = "nv_dnorm",
  family = "normal",
  params = NORM_PARAMS,
  flags = list(log = c(FALSE, TRUE)),
  domain = function(p, f) c(-Inf, Inf),
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_dnorm(anvl::nv_array(x, dtype = dtype), p$mean, p$sd, log = f$log))
  },
  ref_value = function(x, p, f) dnorm(x, mean = p$mean, sd = p$sd, log = f$log),

  grad_wrt = c("x", "mean", "sd"),
  grad = function(x, dtype, p, f) {
    n <- length(x)
    d <- grad_dnorm(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$mean, n), dtype = dtype),
      anvl::nv_array(rep(p$sd, n), dtype = dtype),
      log = f$log
    )
    lapply(list(x = d$x, mean = d$mean, sd = d$sd), as.double)
  },
  ref_grad = dnorm_grad_ref,
  ref_grad_bound_ulp64 = 16,
  ref_grad_mpfr = function(x, p, f) {
    z <- (x - p$mean) / p$sd
    core <- list(x = -z / p$sd, mean = z / p$sd, sd = (z * z - 1) / p$sd)
    if (isTRUE(f$log)) core else lapply(core, function(m) mp_phi_times(z, m / p$sd))
  },

  ## jax.scipy.stats.norm covers both pdf and logpdf, so every variant twins.
  jax_value = function(x, dtype, p, f) {
    jax_init()
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    js <- reticulate::import("jax.scipy.stats", convert = FALSE)
    np <- reticulate::import("numpy")
    fn <- if (isTRUE(f$log)) js$norm$logpdf else js$norm$pdf
    as.double(np$asarray(fn(jnp$asarray(x, dtype = jax_dtype(dtype)), p$mean, p$sd)))
  },
  jax_grad = function(x, dtype, p, f) {
    jax_init()
    reticulate::py_run_string(
      "
import jax
from jax.scipy.stats import norm as _n
def _dnorm_grad(log, i):
    f = _n.logpdf if log else _n.pdf
    return jax.jit(jax.vmap(jax.grad(f, argnums=i), in_axes=(0, None, None)))
"
    )
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    np <- reticulate::import("numpy")
    xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
    g <- function(i) as.double(np$asarray(reticulate::py$`_dnorm_grad`(f$log, i)(xx, p$mean, p$sd)))
    list(x = g(0L), mean = g(1L), sd = g(2L))
  }
)
