## ---------------------------------------------------------------------------
## Shared by the three normal specs. Not a spec itself (leading underscore).
##
##   standard  (mean, sd) = (0, 1), the default. Every affine step is exact, so
##             it isolates the distribution's own mathematics -- and says
##             nothing about the standardisation.
##   shifted   (mean, sd) = (-pi, 2*pi). Neither constant is representable in
##             either precision, so (x - mean)/sd is exercised the way a real
##             caller exercises it. This is also the route that the old
##             literal-precision bug destroyed: `mean` and `sd` arrive as jit
##             arguments, which no anvl-side patch could reach. These cells are
##             what hold that fix in place.
## ---------------------------------------------------------------------------

NORM_PARAMS <- list(
  standard = list(mean = 0, sd = 1),
  shifted = list(mean = -pi, sd = 2 * pi)
)

jax_init <- local({
  done <- FALSE
  function() {
    if (done) {
      return(invisible(TRUE))
    }
    library(reticulate)
    venv <- normalizePath(file.path(here(), "..", "..", "..", "py-benchmarks", ".venv"), mustWork = FALSE)
    if (!nzchar(Sys.getenv("RETICULATE_PYTHON")) && dir.exists(venv)) {
      use_virtualenv(venv, required = TRUE)
    }
    import("jax")$config$update("jax_enable_x64", TRUE)
    done <<- TRUE
    invisible(TRUE)
  }
})

jax_dtype <- function(dtype) {
  jnp <- reticulate::import("jax.numpy", convert = FALSE)
  if (dtype == "f32") jnp$float32 else jnp$float64
}

## z = (x - mean)/sd as an unevaluated sum hi + lo, exact to about 2^-106.
##
## A double z is off by up to half an ulp, and phi(z) = exp(-z^2/2) multiplies
## that by ~z^2: 67 ulp at z = 8.2, ~1500 at z = 38. anvl and base R both
## compute z that way, so a reference that did too would share their
## conditioning error instead of measuring it. hi is the double quotient; lo
## is its exact residual, from an error-free x - mean (TwoSum) and hi * sd
## (Dekker's TwoProduct), divided by sd. For mean = 0, sd = 1 lo is exactly 0.
## Beyond |hi| = 1e150 the split would overflow and lo is taken as 0: phi is 0
## there and every use of z is relatively insensitive to its last ulp.
std_z <- function(x, mean, sd) {
  s <- x - mean
  bb <- s - x
  e <- (x - (s - bb)) + (-mean - bb)
  hi <- s / sd
  split <- function(a) {
    c <- 134217729 * a
    h <- c - (c - a)
    list(h = h, l = a - h)
  }
  lo <- numeric(length(hi))
  ok <- which(is.finite(hi) & abs(hi) < 1e150 & is.finite(e))
  if (length(ok)) {
    a <- split(hi[ok])
    b <- split(rep(sd, length(ok)))
    ph <- hi[ok] * sd
    pl <- ((a$h * b$h - ph) + a$h * b$l + a$l * b$h) + a$l * b$l
    lo[ok] <- (((s[ok] - ph) - pl) + e[ok]) / sd
  }
  list(hi = hi, lo = lo)
}

## The inverse Mills ratio, phi(z) / Phi(z) -- the derivative of log Phi, and
## the quantity every log-scale normal gradient reduces to.
##
## Not exp(dnorm(z, log = TRUE) - pnorm(z, log.p = TRUE)): that subtracts two
## logs of about -z^2/2, and exp() turns their absolute rounding error into a
## relative error of ~|z^2/2| ulp -- 6.9e-14 at z = -50 -- and past z = -1e154
## both logs are -Inf and it is NaN. Instead:
##
##   z >= -20  the direct ratio phi(z)/pnorm(z): both are normal doubles
##             there, phi from phi_times() and pnorm accurate to an ulp or two
##   z <  -20  the continued fraction phi/Phi(-t) = t + 1/(t + 2/(t + ...)),
##             t = -z, evaluated backwards from 40 terms
##
## Both are within an ulp of 300-bit MPFR at the crossover and beyond (checked
## at z = -20 .. -1e3); the fraction needs no special case for huge |z|.
inv_mills <- function(z) {
  out <- rep(NaN, length(z))
  near <- which(!is.na(z) & z >= -20)
  out[near] <- phi_times(z[near], rep(1, length(near))) / pnorm(z[near])
  far <- which(!is.na(z) & z < -20)
  if (length(far)) {
    t <- -z[far]
    v <- t
    for (n in 40:1) {
      v <- t + n / v
    }
    out[far] <- v
  }
  out
}

## m * phi(z + lo), phi the standard normal density, without forming phi first
## and without base R's dnorm(), which below |x| = 5 is a plain
## exp(-0.5 * x * x) whose rounded square costs up to ~z^2/2 ulp (~12 at 5).
##
## z is split as x1 + x2 with x1 on a 2^-16 grid, so x1^2 is exact, and
##   phi(z) = exp(-x1^2/2) * exp(-(x1 + x2/2) x2) / sqrt(2 pi)
## (base R's own tail method, used here everywhere). The first factor is
## evaluated as exp(k - h) * exp(-k), k = floor(h/2), with m applied in
## between, so nothing underflows before the final product: phi underflows at
## |z| ~ 37.5 while m * phi is still representable a little further out. m is
## a factor growing at most polynomially in z (z, z^2 - 1, 1/sd); beyond
## |z| = 40 such m * phi(z) < 2^-1075 and the answer is a signed zero, even
## where m has overflowed. `lo` is the low part of z (see std_z()), folded in
## as exp(-(z + lo/2) lo).
M_1_SQRT_2PI <- 0.398942280401432677939946059934
phi_split <- function(a) {
  x1 <- round(a * 65536) / 65536
  x2 <- a - x1
  h <- 0.5 * x1 * x1
  k <- floor(h / 2)
  list(k = k, h = h, t = (x1 + 0.5 * x2) * x2)
}
phi_times <- function(z, m, lo = 0) {
  ## only where lo is non-zero: at z = +-Inf, Inf * 0 would make it NaN
  lo <- rep_len(lo, length(z))
  j <- which(lo != 0)
  if (length(j)) {
    m[j] <- m[j] * exp(-(z[j] + 0.5 * lo[j]) * lo[j])
  }
  out <- 0 * sign(m)
  i <- which(!is.na(z) & abs(z) <= 40)
  if (length(i)) {
    p <- phi_split(abs(z[i]))
    ## m is a factor of ordinary size here (z, z^2 - 1, 1/sd), never subnormal
    out[i] <- ((m[i] * M_1_SQRT_2PI) * exp(p$k - p$h) * exp(-p$t)) * exp(-p$k)
  }
  out[is.na(z) | is.na(m)] <- NaN
  out
}

## m / phi(z + lo), the reciprocal counterpart, by the same split: it overflows only
## when the result truly exceeds the double range. Beyond |z| = 40 it is
## +-Inf for any m of ordinary size.
phi_recip <- function(z, m, lo = 0) {
  ## 1/phi(z + lo) = exp((z + lo/2) lo) / phi(z), folded into m where lo != 0
  lo <- rep_len(lo, length(z))
  j <- which(lo != 0)
  if (length(j)) {
    m[j] <- m[j] * exp((z[j] + 0.5 * lo[j]) * lo[j])
  }
  out <- sign(m) * Inf
  i <- which(!is.na(z) & abs(z) <= 40)
  if (length(i)) {
    p <- phi_split(abs(z[i]))
    ## m first times the large factor: m may be a subnormal probability, and
    ## dividing it by 1/sqrt(2 pi) first would round it to a few bits
    out[i] <- (((m[i] * exp(p$h - p$k)) * exp(p$t)) / M_1_SQRT_2PI) * exp(p$k)
  }
  out[is.na(z) | is.na(m)] <- NaN
  out
}

## The standard normal quantile of a probability, as hi + lo.
##
## hi is base R's qnorm() -- in the far tails refined by Newton, since at the
## smallest subnormal p it is ~5e-3 out -- a few ulp from the true quantile; that is enough to
## ruin any derivative evaluated at it, because 1/phi(z) turns an error dz in
## z into a relative error |z| dz: ~300 ulp at p = 1e-100. lo is one Newton
## step taken in whichever tail is small -- so that its target probability is
## formed without cancellation (p, 1 - p, exp(log p) or -expm1(log p), each
## exact or accurate; a log target only where log p < -1) and compared with an R tail function accurate to a few ulp
## *relative*:
##
##   lower tail of z, target t:      lo = (t - Phi(z)) / phi(z)
##   upper tail of z, target t:      lo = (Q(z) - t) / phi(z)
##   lower tail, target log t:       lo = (log t - log Phi(z)) / m(z)
##   upper tail, target log t:       lo = (log Q(z) - log t) / m(-z)
##
## with m the inverse Mills ratio. A tail probability good to k ulp gives lo
## good to ~k eps / |z|, so |z| lo -- what the derivative sees -- is good to a
## few ulp. Where phi(z) is subnormal, or z is infinite, lo is 0: 1/phi is
## beyond the double range there anyway.
std_q <- function(pr, lower, log_p) {
  hi <- suppressWarnings(qnorm(pr, 0, 1, lower.tail = lower, log.p = log_p))
  lo <- numeric(length(hi))
  ok <- !is.na(hi) & is.finite(hi)
  ph <- rep(NA_real_, length(hi))
  ph[ok] <- phi_times(hi[ok], rep(1, sum(ok)))
  ## the tail of z that is small, and its target
  if (!log_p) {
    small <- pr <= 0.5
    t <- ifelse(small, pr, 1 - pr) # 1 - pr is exact for pr >= 1/2
    lower_z <- if (lower) small else !small
    i <- which(ok & ph >= 2^-1022)
    li <- i[lower_z[i]]
    ui <- i[!lower_z[i]]
    lo[li] <- (t[li] - pnorm(hi[li])) / ph[li]
    lo[ui] <- (pnorm(hi[ui], lower.tail = FALSE) - t[ui]) / ph[ui]
  } else {
    ## Far out (log p < -1) the target stays a log. Nearer the centre it is a
    ## probability: exp(log p) or -expm1(log p), both accurate -- a log target
    ## there would compare log p with log Phi(0), which is the same rounded
    ## log(1/2), and lose z entirely where it is ~1e-17.
    far <- pr < -1
    lower_z <- if (lower) (far | pr <= -log(2)) else !(far | pr <= -log(2))
    t <- ifelse(pr <= -log(2), exp(pr), -expm1(pr))
    i <- which(ok)
    ll <- i[far[i] & lower_z[i]]
    lu <- i[far[i] & !lower_z[i]]
    lo[ll] <- (pr[ll] - pnorm(hi[ll], log.p = TRUE)) / inv_mills(hi[ll])
    lo[lu] <- (pnorm(hi[lu], lower.tail = FALSE, log.p = TRUE) - pr[lu]) / inv_mills(-hi[lu])
    j <- i[!far[i] & ph[i] >= 2^-1022]
    pl <- j[lower_z[j]]
    pu <- j[!lower_z[j]]
    lo[pl] <- (t[pl] - pnorm(hi[pl])) / ph[pl]
    lo[pu] <- (pnorm(hi[pu], lower.tail = FALSE) - t[pu]) / ph[pu]
  }
  ## At the centre both tails are ~1/2 and the difference that fixes z is
  ## lost in either: work with p - 1/2 and Phi(z) - 1/2 directly. p - 1/2 is
  ## exact on the probability scale; from log p it is expm1(log p + log 2)/2
  ## with log 2 split so that the sum is exact (Sterbenz). Phi(z) - 1/2 is its
  ## Taylor series, converged to double for |z| < 0.7 by 20 terms.
  cen <- which(ok & (if (log_p) abs(pr + log(2)) < 0.4 else abs(pr - 0.5) < 0.25))
  if (length(cen)) {
    d <- if (log_p) 0.5 * expm1((pr[cen] + LN2_HI) + LN2_LO) else pr[cen] - 0.5
    if (!lower) {
      d <- -d
    }
    lo[cen] <- (d - phi_half(hi[cen])) / ph[cen]
  }
  ## In the far tails the probability target is corrected without R's pnorm,
  ## whose result there can be subnormal and keep only a few bits: with
  ## Phi(z) = phi(z)/m(z) (m the inverse Mills ratio, from its continued
  ## fraction) the step is lo = t/phi(z) - 1/m(z), and t/phi comes from the
  ## scaled phi_recip() -- each accurate even for a subnormal target t.
  if (!log_p) {
    far_t <- which(ok & abs(hi) > 20)
  } else {
    far_t <- which(ok & pr >= -1 & abs(hi) > 20)
  }
  if (length(far_t)) {
    zf <- hi[far_t]
    tf <- if (log_p) {
      ifelse(pr[far_t] <= -log(2), exp(pr[far_t]), -expm1(pr[far_t]))
    } else {
      ifelse(pr[far_t] <= 0.5, pr[far_t], 1 - pr[far_t])
    }
    low <- zf < 0 # the small tail of z is its lower tail
    step <- function(z) ifelse(low, phi_recip(z, tf) - 1 / inv_mills(z), 1 / inv_mills(-z) - phi_recip(z, tf))
    ## base R's qnorm() is itself ~5e-3 out at the smallest subnormal p, too
    ## far for one step: step until it settles, folding each into hi, and keep
    ## the last step, taken at the final hi, as lo
    for (k in 1:4) {
      d <- step(zf)
      big <- is.finite(d) & abs(d) > 1e-12 * abs(zf)
      if (!any(big)) {
        break
      }
      zf[big] <- zf[big] + d[big]
    }
    hi[far_t] <- zf
    lo[far_t] <- step(zf)
  }
  lo[!is.finite(lo)] <- 0
  list(hi = hi, lo = lo)
}

LN2_HI <- 0.6931471805599453
LN2_LO <- 2.3190468138462996e-17

## Phi(z) - 1/2 = phi(0) sum_n (-1)^n z^(2n+1) / (2^n n! (2n+1)), for |z| < 0.7
phi_half <- function(z) {
  z2 <- z * z
  term <- z
  acc <- z
  for (n in 1:20) {
    term <- -term * z2 / (2 * n)
    acc <- acc + term / (2 * n + 1)
  }
  M_1_SQRT_2PI * acc
}

## ---- high-precision truths, for validate-refs only -------------------------
##
## Rmpfr is needed here and nowhere else; a sweep never calls these. Each
## evaluates the mathematics in MPFR at the precision of its arguments, and
## conditions are tested on the exact doubles, which are exact.

mp_phi <- function(z) exp(-z * z / 2) / sqrt(2 * Rmpfr::Const("pi", mp_prec(z)))

## m * phi(z); where z is infinite phi is 0 and the answer is a zero signed
## like m, not Inf * 0. Beyond MPFR's exponent range exp() is 0, which is the
## true double answer there.
mp_phi_times <- function(z, m) {
  out <- m * mp_phi(z)
  zi <- which(is.infinite(mp_num(z)))
  if (length(zi)) {
    out[zi] <- Rmpfr::mpfr(0 * sign(mp_num(m[zi])), mp_prec(z))
  }
  out
}

## phi(z)/Phi(z): the direct ratio above z = -30, and below it the continued
## fraction t + 1/(t + 2/(t + ...)), t = -z, from 120 terms (converged far
## beyond double precision for t >= 30) -- in MPFR, where
## the fraction needs no exp() and so works for any z the double line holds.
mp_imills <- function(z, terms = 120L) {
  zn <- mp_num(z)
  out <- z
  near <- which(!is.na(zn) & zn >= -30)
  if (length(near)) {
    out[near] <- mp_phi(z[near]) / Rmpfr::pnorm(z[near])
  }
  far <- which(!is.na(zn) & zn < -30)
  if (length(far)) {
    t <- -z[far]
    v <- t
    for (n in terms:1) {
      v <- t + n / v
    }
    out[far] <- v
  }
  out
}

## log Phi(z), without exp() below z = -30: log phi(z) - log(phi/Phi); and
## as log1p(-Q(z)) above 0, since at 256 bits Phi(20) is exactly 1.
mp_log_Phi <- function(z) {
  zn <- mp_num(z)
  out <- z
  near <- which(!is.na(zn) & zn >= -30 & zn <= 0)
  if (length(near)) {
    out[near] <- log(Rmpfr::pnorm(z[near]))
  }
  pos <- which(!is.na(zn) & zn > 0)
  if (length(pos)) {
    out[pos] <- log1p(-Rmpfr::pnorm(-z[pos]))
  }
  far <- which(!is.na(zn) & zn < -30)
  if (length(far)) {
    zf <- z[far]
    out[far] <- -zf * zf / 2 - log(sqrt(2 * Rmpfr::Const("pi", mp_prec(z)))) - log(mp_imills(zf))
  }
  out
}

## The standard normal quantile of the exact binary probability `pr`, by
## Newton's method in MPFR from base R's double, on the scale given: Phi(z) = p,
## Q(z) = p, log Phi(z) = log p or log Q(z) = log p. Q(z) = Phi(-z).
mp_qnorm <- function(pr, lower, log_p, prec) {
  hi <- suppressWarnings(qnorm(pr, lower.tail = lower, log.p = log_p))
  z <- Rmpfr::mpfr(hi, prec)
  ok <- which(is.finite(hi))
  if (!length(ok)) {
    return(z)
  }
  s <- if (lower) 1 else -1
  zk <- z[ok]
  tgt <- Rmpfr::mpfr(pr[ok], prec)
  ## four steps from a start good to ~1e-16: quadratic convergence leaves the
  ## quantile good to far below double precision
  for (it in 1:4) {
    zs <- s * zk
    if (!log_p) {
      zk <- zk - (Rmpfr::pnorm(zs) - tgt) / (s * mp_phi(zs))
    } else {
      m <- mp_imills(zs)
      zn <- mp_num(zs)
      lcdf <- zs
      far <- which(zn < -30)
      nr <- which(zn >= -30)
      if (length(far)) {
        lcdf[far] <- -zs[far] * zs[far] / 2 - log(sqrt(2 * Rmpfr::Const("pi", prec))) - log(m[far])
      }
      if (length(nr)) {
        lcdf[nr] <- mp_log_Phi(zs[nr])
      }
      zk <- zk - (lcdf - tgt) / (s * m)
    }
  }
  z[ok] <- zk
  z
}
