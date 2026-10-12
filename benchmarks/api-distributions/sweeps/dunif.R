## ---------------------------------------------------------------------------
## nv_dunif against base R dunif().
##
## The reference is base R at the same precision, so this sweep can only see
## errors larger than f64 rounding of dunif() itself. It is a search for
## structural breakage -- a wrong branch, a NaN leak, a flushed subnormal --
## not a sub-ulp accuracy audit. In f32 the reference is computed in f64 and
## rounded, so there the sweep does bound the rounding too.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_uniform.R"), local = TRUE)

## With the limits fixed, finite and non-degenerate (w = max - min > 0) the
## density is piecewise constant in `x`, so
##
##   d/dx     0                                            everywhere
##   d/dmin   +1/w^2 (log = FALSE)  or  +1/w (log = TRUE)   on the support
##   d/dmax   -1/w^2 (log = FALSE)  or  -1/w (log = TRUE)   on the support
##
## and 0 off the support, where the density is the constant 0 (resp. -Inf) and
## moving a limit does not change it. The support is closed -- x == min and
## x == max are in it -- matching the (x >= min) & (x <= max) indicator in
## nv_dunif(); at those two kinks the derivative does not exist and this is a
## convention, not a claim.
dunif_grad_ref <- function(x, p, f) {
  w <- p$max - p$min
  on <- !is.nan(x) & x >= p$min & x <= p$max
  d <- list(
    x = 0,
    min = if (isTRUE(f$log)) 1 / w else 1 / w^2,
    max = if (isTRUE(f$log)) -1 / w else -1 / w^2
  )
  lapply(d, function(z) ifelse(is.nan(x), NaN, ifelse(on, z, 0)))
}

## The limits are passed as full-length arrays rather than scalars so that each
## element carries its own d/dmin and d/dmax; with scalar limits the gradient
## would be one summed number and there would be nothing to sweep.
grad_dunif <- anvl::jit(
  anvl::gradient(
    \(x, min, max, log = FALSE) sum(anvl::nv_dunif(x, min, max, log = log)),
    wrt = c("x", "min", "max")
  ),
  static = "log"
)

sweep_spec(
  name = "nv_dunif",
  family = "uniform",
  params = UNIF_INTERVALS,
  flags = list(log = c(FALSE, TRUE)),

  ## Defined on the whole line. Off the support the density is a constant (0,
  ## or -Inf on the log scale), so a disagreement there is a real finding, not
  ## a domain artefact: the support is reported, and excuses nothing.
  domain = function(p, f) c(-Inf, Inf),
  support = function(p, f) c(p$min, p$max),
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_dunif(anvl::nv_array(x, dtype = dtype), p$min, p$max, log = f$log))
  },
  ref_value = function(x, p, f) dunif(x, min = p$min, max = p$max, log = f$log),

  grad_wrt = c("x", "min", "max"),
  grad = function(x, dtype, p, f) {
    n <- length(x)
    d <- grad_dunif(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$min, n), dtype = dtype),
      anvl::nv_array(rep(p$max, n), dtype = dtype),
      log = f$log
    )
    lapply(list(x = d$x, min = d$min, max = d$max), as.double)
  },
  ref_grad = dunif_grad_ref,
  ref_grad_bound_ulp64 = 4,
  ref_grad_mpfr = function(x, p, f) {
    xn <- mp_num(x)
    w <- p$max - p$min
    on <- !is.nan(xn) & xn >= mp_num(p$min) & xn <= mp_num(p$max)
    zero <- mp_fill(x, 0)
    d <- list(
      x = zero,
      min = zero + (if (isTRUE(f$log)) 1 / w else 1 / (w * w)),
      max = zero + (if (isTRUE(f$log)) -1 / w else -1 / (w * w))
    )
    lapply(d, function(v) {
      v[!on] <- 0
      v[is.nan(xn)] <- NaN
      v
    })
  },

  ## jax.scipy.stats.uniform is parameterised by (loc, scale) = (min, max - min)
  ## and covers both the density and its log, so every variant has a twin. The
  ## wrapper restates it in anvl's (min, max) parameterisation *before*
  ## differentiating -- otherwise d/dscale would be compared against d/dmax,
  ## which differ by the d/dmin term of the chain rule.
  jax_value = function(x, dtype, p, f) {
    jax_init()
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    js <- reticulate::import("jax.scipy.stats", convert = FALSE)
    np <- reticulate::import("numpy")
    fn <- if (isTRUE(f$log)) js$uniform$logpdf else js$uniform$pdf
    as.double(np$asarray(fn(jnp$asarray(x, dtype = jax_dtype(dtype)), p$min, p$max - p$min)))
  },
  jax_grad = function(x, dtype, p, f) {
    jax_init()
    reticulate::py_run_string(
      "
import jax
from jax.scipy.stats import uniform as _u
def _dunif_grad(log, i):
    f = _u.logpdf if log else _u.pdf
    g = jax.grad(lambda x, mn, mx: f(x, mn, mx - mn), argnums=i)
    return jax.jit(jax.vmap(g, in_axes=(0, None, None)))
"
    )
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    np <- reticulate::import("numpy")
    xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
    g <- function(i) {
      as.double(np$asarray(reticulate::py$`_dunif_grad`(f$log, i)(xx, p$min, p$max)))
    }
    list(x = g(0L), min = g(1L), max = g(2L))
  }
)
