## ---------------------------------------------------------------------------
## A synthetic family that exercises the contract, with no anvl in sight.
##
## It exists because a sweep that silently sweeps nothing is indistinguishable
## from a sweep that found nothing, and at full depth a cell costs minutes --
## far too expensive a way to discover that a filter was misspelled or a
## reducer stopped reducing. Every mechanism the real specs rely on is used
## here, and the errors are injected at known places so the engine has to find
## them.
##
##   value/clean    reference reproduced exactly            -> expect 0
##   value/nudged   one ulp added on a known interval       -> expect ~1 ulp
##   grad           two outputs from one pass, one broken   -> expect a FAIL
##   ranges         a deliberate NaN-only disagreement      -> expect "nan"
##   value/pinhole  right everywhere but x = 1 exactly     -> only the exact
##                                                             points see it in f64
##   value/weakref  the *reference* is wrong on (0, 1e-20)  -> candidate base R
##                  and a stable reference is right          disputes, nothing else
##
## Run it with:  Rscript run.R selftest
## ---------------------------------------------------------------------------

nudge <- function(x, lo, hi) {
  ## add one ulp, but only on [lo, hi], so the sweep has to locate the interval
  i <- !is.na(x) & x >= lo & x <= hi
  x[i] <- x[i] + ulp_size(x[i], "f64")
  x
}

sweep_spec(
  name = "selftest",
  family = "_selftest",
  dtypes = c("f32", "f64"),
  params = list(
    clean = list(err = 0),
    nudged = list(err = 1),
    pinhole = list(err = -1),
    weakref = list(err = -2)
  ),
  flags = list(broken = c(FALSE, TRUE)),

  ## Defined on the whole line, so no failure here can be excused as lying
  ## outside the domain: every region found must carry a real cause.
  domain = function(params, flags) c(-Inf, Inf),
  value = function(x, dtype, params, flags) {
    y <- abs(x)
    if (params$err > 0) {
      y <- nudge(y, 1, 2)
    }
    if (params$err == -1) {
      y[x %in% 1] <- NaN
    }
    if (isTRUE(flags$broken)) {
      y[!is.na(x) & x > 1e300] <- 0
    }
    if (dtype == "f32") as_f32(y) else y
  },
  ref_value = function(x, params, flags) {
    y <- abs(x)
    if (params$err == -2) {
      ## a reference that is off by 1e-10 on (0, 1e-30) and loses the value
      ## entirely on [1e-30, 1e-20): finite and non-finite disputes
      i <- !is.na(x) & x > 0 & x < 1e-30
      y[i] <- y[i] * (1 + 1e-10)
      y[!is.na(x) & x >= 1e-30 & x < 1e-20] <- 0
    }
    y
  },
  ref_stable = function(x, params, flags) abs(x),
  ref_stable_mpfr = function(x, params, flags) abs(x),
  ref_stable_bound_ulp64 = 0,
  ref_stable_note = "the selftest's weakref reference is deliberately wrong on (0, 1e-20)",

  ## Two outputs from one call, which is the mechanism the gradient cells use.
  grad_wrt = c("x", "scale"),
  grad = function(x, dtype, params, flags) {
    g <- list(x = sign(x), scale = abs(x))
    if (isTRUE(flags$broken)) {
      g$scale <- g$scale * (1 + 1e-6)
    }
    if (dtype == "f32") lapply(g, as_f32) else g
  },
  ref_grad = function(x, params, flags) {
    list(x = sign(x), scale = abs(x))
  },
  ref_grad_bound_ulp64 = 0,
  ref_grad_mpfr = function(x, params, flags) {
    ## the sign from the exact doubles: Rmpfr's sign() does not keep NaN
    xn <- mp_num(x)
    list(x = Rmpfr::mpfr(sign(xn), mp_prec(x)), scale = abs(x))
  }
)
