# The HMC sampler of hmc-anvl.R, written in R around TMB's compiled objective
# (the potential energy) and its gradient.
hmc_tmb <- function(obj, beta, eps, L, n_draws) {
  p <- length(beta)
  draws <- matrix(NA_real_, n_draws, p)
  grad <- function(b) as.vector(obj$gr(b))
  for (i in seq_len(n_draws)) {
    rho <- rnorm(p)
    b <- beta
    r <- rho
    g <- grad(b)
    for (l in seq_len(L)) {
      r <- r - eps / 2 * g
      b <- b + eps * r
      g <- grad(b)
      r <- r - eps / 2 * g
    }
    H0 <- obj$fn(beta) + sum(rho^2) / 2
    H1 <- obj$fn(b) + sum(r^2) / 2
    if (log(runif(1L)) < H0 - H1) beta <- b
    draws[i, ] <- beta
  }
  draws
}
