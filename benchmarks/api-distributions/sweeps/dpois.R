## ---------------------------------------------------------------------------
## nv_dpois against base R dpois(), swept in lambda (see _poisson.R for why).
##
## nv_dpois() evaluates Loader's saddle point expansion, as base R does, with
## the deviance term bd0(x, lambda) taken as its series near x = lambda and
## formed from x - lambda. Base R's dpois_raw() loses accuracy growing with x at a
## full-mantissa lambda: up to ~1.6e4 ulps of the log density and ~2.5e5 of the
## density at x = 1e6. JAX's
## logpmf differences xlogy(x, lambda) and lgamma(x + 1), each of size
## ~x log x, which costs it most at the `large` x. The stable reference below is
## what lets the sweep tell base R's error from anvl's.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_poisson.R"), local = TRUE)

## With f the density, d/dlambda f = f (x - lambda) / lambda, and on the log
## scale (x - lambda) / lambda. The density-scale form is evaluated as
##
##   lambda >= x:  f(x) ((x - lambda) / lambda)
##   lambda <  x:  f(x - 1) ((x - lambda) / x)        (f(x) x / lambda = f(x - 1))
##
## with f from the stable log density (pois_log_density(), _poisson.R), so that
## the factor beside the density is at most 1 in size and the product
## underflows only where the answer does: f(x) itself has underflowed well
## before f(x) x / lambda when lambda is tiny. At x = 0 it is -exp(-lambda), on
## the log scale -1.
##
## At the ends of [0, Inf], the limits:
##
##   lambda = 0:    f' = -1 at x = 0, 1 at x = 1, 0 otherwise
##                  log:  -1 at x = 0, +Inf otherwise
##   lambda = Inf:  f' = 0;  log: -1
##
## anvl returns 0 at lambda = 0 for x >= 1 and at lambda = Inf, documented
## conventions; the reference keeps the mathematical limit, so those
## disagreements are recorded at the domain endpoints as boundary behaviour.
## Below 0 the density is NaN and every derivative is 0.
dpois_grad_ref <- function(lam, p, f) {
  x <- p$x
  log <- isTRUE(f$log)
  out <- rep(0, length(lam))
  out[is.na(lam)] <- NaN
  i <- which(!is.na(lam) & lam > 0 & lam < Inf)
  if (length(i)) {
    l <- lam[i]
    out[i] <- if (log) {
      (x - l) / l
    } else if (x == 0) {
      -exp(-l)
    } else {
      ifelse(
        l >= x,
        exp(pois_log_density(x, l)) * ((x - l) / l),
        exp(pois_log_density(x - 1, l)) * ((x - l) / x)
      )
    }
  }
  at0 <- which(!is.na(lam) & lam == 0)
  out[at0] <- if (x == 0) {
    -1
  } else if (log) {
    Inf
  } else {
    as.numeric(x == 1)
  }
  atinf <- which(!is.na(lam) & lam == Inf)
  out[atinf] <- if (log) -1 else 0
  list(lambda = out)
}

grad_dpois <- anvl::jit(
  anvl::gradient(
    \(x, lambda, log = FALSE) sum(anvl::nv_dpois(x, lambda, log = log)),
    wrt = "lambda"
  ),
  static = "log"
)

sweep_spec(
  name = "nv_dpois",
  family = "poisson",
  params = POIS_D_PARAMS,
  flags = list(log = c(FALSE, TRUE)),

  ## The swept argument is lambda. Below 0 base R returns NaN with a warning;
  ## a value disagreement there is still a failure.
  domain = function(p, f) c(0, Inf),
  support = function(p, f) c(0, Inf),
  ## bd0(x, lambda) switches from its direct form to its series where
  ## |x - lambda| < 0.1 (x + lambda), i.e. between lambda = 9 x / 11 and
  ## 11 x / 9; log_ratio(lambda, x) rescales where lambda / x falls below the
  ## smallest normal; and bd0 scales its operands down where the larger is
  ## above 2^120 (f32) or 2^1016 (f64). x = 0 takes none of them. See
  ## R/api-distributions.R, bd0, log_ratio and nv_dpois.
  branch_points = function(p, f, dtype) {
    x <- p$x
    if (x > 0) {
      f32 <- dtype == "f32"
      c(
        bd0_series_below = 9 * x / 11,
        bd0_series_above = 11 * x / 9,
        ratio_underflow = x * (if (f32) 2^-126 else 2^-1022),
        bd0_scaled = if (f32) 2^120 else 2^1016
      )
    }
  },
  ## x is the operand that sets the precision, so it is passed as an array of
  ## the cell's dtype; lambda is the swept array.
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_dpois(
      anvl::nv_scalar(p$x, dtype = dtype),
      anvl::nv_array(x, dtype = dtype),
      log = f$log
    ))
  },
  ref_value = function(x, p, f) suppressWarnings(dpois(p$x, x, log = f$log)),
  ref_stable = pois_density_stable,
  ref_stable_mpfr = function(x, p, f) {
    lam <- mp_num(x)
    xn <- mp_num(p$x)
    log <- isTRUE(f$log)
    ld <- p$x * log(x) - x - lgamma(p$x + 1)
    out <- if (log) ld else exp(ld)
    ## the ends of [0, Inf], where 0 log 0 is 0 and the logs are -Inf
    out[!is.nan(lam) & lam == 0] <- if (xn == 0) (if (log) 0 else 1) else (if (log) -Inf else 0)
    out[!is.nan(lam) & lam == Inf] <- if (log) -Inf else 0
    out[is.nan(lam) | lam < 0] <- NaN
    out
  },
  ## As nv_dbinom()'s: the log density is a sum of terms each up to about
  ## |log f| + 10 in size (the deviance term, the Stirling correction and
  ## log(2 pi x) / 2), and each operation rounds at their scale, an absolute
  ## error of a few ulp(|log f| + 10); exp() turns that into up to a few
  ## thousand f64 ulps of the density near the bottom of the double range, where
  ## |log f| ~ 708. 4096 leaves margin over that; base R's error at x = 1e6 is
  ## up to ~1.6e4 ulps on the log scale and ~2.5e5 on the density.
  ref_stable_bound_ulp64 = 4096,
  ref_stable_note = "base R's dpois_raw() loses accuracy growing with x at a lambda with a full mantissa: up to ~1.6e4 ulps of the log density and ~2.5e5 of the density at x = 1e6, ~60 and ~260 at x = 1000",

  grad_wrt = "lambda",
  grad = function(x, dtype, p, f) {
    d <- grad_dpois(
      anvl::nv_scalar(p$x, dtype = dtype),
      anvl::nv_array(x, dtype = dtype),
      log = f$log
    )
    list(lambda = as.double(d$lambda))
  },
  ref_grad = dpois_grad_ref,
  ## As the stable density: the density-scale derivative carries its error,
  ## plus ~3 roundings for the factor and the product; the log-scale one is two
  ## roundings. 4096 covers the former.
  ref_grad_bound_ulp64 = 4096,
  ref_grad_mpfr = function(x, p, f) {
    lam <- mp_num(x)
    xn <- mp_num(p$x)
    log <- isTRUE(f$log)
    d <- p$x - x
    out <- if (log) d / x else exp(p$x * log(x) - x - lgamma(p$x + 1)) * d / x
    out[!is.nan(lam) & lam == 0] <- if (xn == 0) {
      -1
    } else if (log) {
      Inf
    } else {
      as.numeric(xn == 1)
    }
    out[!is.nan(lam) & lam == Inf] <- if (log) -1 else 0
    out[!is.nan(lam) & lam < 0] <- 0
    out[is.nan(lam)] <- NaN
    list(lambda = out)
  },

  ## jax.scipy.stats.poisson has pmf and logpmf, so every variant has a twin.
  jax_value = function(x, dtype, p, f) jax_dpois_value(x, dtype, p, isTRUE(f$log)),
  jax_grad = function(x, dtype, p, f) jax_dpois_grad(x, dtype, p, isTRUE(f$log))
)
