## ---------------------------------------------------------------------------
## nv_pnorm against base R pnorm().
##
## nv_pnorm carries two symmetric tail branches -- an asymptotic Mills-ratio
## series in the far lower tail and its mirror image near 1 -- and the sweep
## crosses both seams from both sides. That symmetry is the point: every
## accuracy grid this function was originally developed against was one-sided
## (deeply negative z only), which left the upper-tail branch unexercised in
## the artifact even though unit tests happened to cover it. A bit-pattern
## sweep has no sidedness to get wrong.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_normal.R"), local = TRUE)

## With z = (q - mean)/sd and phi the standard density, the lower-tail CDF has
##
##   d/dq  phi(z)/sd     d/dmean  -phi(z)/sd     d/dsd  -z phi(z)/sd
##
## and the upper tail flips every sign. On the log scale each is divided by the
## probability itself; that ratio is formed in log space -- exp(log phi - log P)
## rather than phi/P -- because in the far tail both underflow while their
## ratio stays perfectly ordinary, which is the whole reason the log variants
## exist.
pnorm_grad_ref <- function(q, p, f) {
  zz <- std_z(q, p$mean, p$sd) # z as hi + lo; see std_z() in _normal.R
  z <- zz$hi
  zl <- zz$lo
  lower <- isTRUE(f$lower_tail)
  s <- if (lower) 1 else -1
  if (isTRUE(f$log_p)) {
    ## On the log scale the ratio is the inverse Mills ratio, taken at z for
    ## the lower tail and at -z for the upper by symmetry. See inv_mills().
    ## With m = inv_mills and z + lo the full z (sign-flipped for the upper
    ## tail), lo enters to first order.
    zs <- if (lower) z else -z
    ls <- if (lower) zl else -zl
    m0 <- inv_mills(zs)
    ## Far lower tail (zs < -20): m is large and ordinary, and
    ## m(z + lo) = m(z) (1 - (z + m) lo), from m' = -m (z + m).
    ## (the correction only where lo is non-zero: at z = -Inf, -Inf + Inf is NaN)
    r <- ifelse(ls == 0, m0, m0 * (1 - (zs + m0) * ls)) / p$sd
    zr <- (z + zl) * r
    ## Elsewhere phi / Phi with Phi(z + lo) = Phi(z) (1 + m lo), and every
    ## multiplier folded into phi_times(): far into the other tail phi / Phi is
    ## subnormal, and multiplying it by z afterwards would scale up its
    ## rounding, where folding z in rounds once.
    i <- which(!is.na(zs) & zs >= -20)
    if (length(i)) {
      den <- pnorm(zs[i])
      c <- 1 / (p$sd * ifelse(ls[i] == 0, 1, 1 + m0[i] * ls[i]))
      r[i] <- phi_times(zs[i], c, ls[i]) / den
      zr[i] <- phi_times(zs[i], (z[i] + zl[i]) * c, ls[i]) / den
    }
    return(list(q = s * r, mean = -s * r, sd = -s * zr))
  }
  ## On the probability scale through phi_times(): phi underflows near
  ## |z| = 37.5 while z * phi(z) / sd is still representable (-4.4e-323 at
  ## q = 38.6 for d/dsd), and a product formed after the underflow is 0.
  r <- phi_times(z, rep(1 / p$sd, length(z)), zl)
  list(q = s * r, mean = -s * r, sd = -s * phi_times(z, (z + zl) / p$sd, zl))
}

grad_pnorm <- anvl::jit(
  anvl::gradient(
    \(q, mean, sd, lower_tail = TRUE, log_p = FALSE) {
      sum(anvl::nv_pnorm(q, mean, sd, lower_tail = lower_tail, log_p = log_p))
    },
    wrt = c("q", "mean", "sd")
  ),
  static = c("lower_tail", "log_p")
)

sweep_spec(
  name = "nv_pnorm",
  family = "normal",
  params = NORM_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),
  domain = function(p, f) c(-Inf, Inf),
  ## nv_pnorm branches on d = (q - mean) / sd (or (mean - q) / sd for the upper
  ## tail): the asymptotic series at d <= -11.9 in f32 / -20 in f64, the direct
  ## erfc below 0, the complement above. See R/api-distributions.R, nv_pnorm.
  branch_points = function(p, f, dtype) {
    d <- c(asymptotic = if (dtype == "f32") -11.9 else -20, erfc_split = 0)
    if (isTRUE(f$lower_tail)) p$mean + p$sd * d else p$mean - p$sd * d
  },
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_pnorm(
      anvl::nv_array(x, dtype = dtype),
      p$mean,
      p$sd,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    pnorm(x, mean = p$mean, sd = p$sd, lower.tail = f$lower_tail, log.p = f$log_p)
  },

  grad_wrt = c("q", "mean", "sd"),
  grad = function(x, dtype, p, f) {
    n <- length(x)
    d <- grad_pnorm(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$mean, n), dtype = dtype),
      anvl::nv_array(rep(p$sd, n), dtype = dtype),
      lower_tail = f$lower_tail,
      log_p = f$log_p
    )
    lapply(list(q = d$q, mean = d$mean, sd = d$sd), as.double)
  },
  ref_grad = pnorm_grad_ref,
  ref_grad_bound_ulp64 = 16,
  ref_grad_mpfr = function(x, p, f) {
    z <- (x - p$mean) / p$sd
    lower <- isTRUE(f$lower_tail)
    s <- if (lower) 1 else -1
    zs <- s * z
    if (isTRUE(f$log_p)) {
      ## r = phi/Phi at zs; z r formed with z folded in, so it is 0, not
      ## Inf * 0, where zs = +Inf
      r <- mp_imills(zs) / p$sd
      zn <- mp_num(zs)
      zr <- z * r
      near <- which(!is.na(zn) & zn >= -30)
      if (length(near)) zr[near] <- mp_phi_times(zs[near], z[near] / p$sd) / Rmpfr::pnorm(zs[near])
    } else {
      r <- mp_phi_times(z, mp_fill(z, 1) / p$sd)
      zr <- mp_phi_times(z, z / p$sd)
    }
    list(q = s * r, mean = -s * r, sd = -s * zr)
  },

  ## jax.scipy.stats.norm has cdf, logcdf, sf and logsf, so all four variants
  ## have a twin -- the only distribution here where that is true.
  jax_value = function(x, dtype, p, f) {
    jax_init()
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    js <- reticulate::import("jax.scipy.stats", convert = FALSE)
    np <- reticulate::import("numpy")
    fn <- if (isTRUE(f$lower_tail)) {
      if (isTRUE(f$log_p)) js$norm$logcdf else js$norm$cdf
    } else {
      if (isTRUE(f$log_p)) js$norm$logsf else js$norm$sf
    }
    as.double(np$asarray(fn(jnp$asarray(x, dtype = jax_dtype(dtype)), p$mean, p$sd)))
  },
  jax_grad = function(x, dtype, p, f) {
    jax_init()
    reticulate::py_run_string(
      "
import jax
from jax.scipy.stats import norm as _n
def _pnorm_grad(lower, logp, i):
    f = (_n.logcdf if logp else _n.cdf) if lower else (_n.logsf if logp else _n.sf)
    return jax.jit(jax.vmap(jax.grad(f, argnums=i), in_axes=(0, None, None)))
"
    )
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    np <- reticulate::import("numpy")
    xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
    g <- function(i) {
      as.double(np$asarray(
        reticulate::py$`_pnorm_grad`(f$lower_tail, f$log_p, i)(xx, p$mean, p$sd)
      ))
    }
    list(q = g(0L), mean = g(1L), sd = g(2L))
  }
)
