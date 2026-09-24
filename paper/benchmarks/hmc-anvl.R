# Hamiltonian Monte Carlo for Bayesian logistic regression, written with anvl.
# The labelled sections are read into the paper via knitr::read_chunk(), so the
# code shown in the paper is exactly the code that is benchmarked.

## ---- potential
potential <- function(beta, X, y) {
  eta <- X %*% t(beta)
  y <- nv_broadcast_to(y, shape(eta))
  loglik <- nv_sum(y * eta - nv_log1p(exp(eta)), axes = 1L)
  logprior <- -nv_sum(beta^2, axes = 2L) / 2
  -(loglik + logprior)
}

## ---- gradient
grad_potential <- gradient(
  function(beta, X, y) sum(potential(beta, X, y)),
  wrt = "beta"
)

## ---- leapfrog
leapfrog <- function(beta, p, X, y, eps, L) {
  init <- list(beta = beta, p = p, grad = grad_potential(beta, X, y)$beta)
  nv_scan(init, steps = L, body = function(s, x) {
    p <- s$p - eps / 2 * s$grad
    beta <- s$beta + eps * p
    grad <- grad_potential(beta, X, y)$beta
    p <- p - eps / 2 * grad
    list(carry = list(beta = beta, p = p, grad = grad), out = NULL)
  })$carry
}

## ---- hmc-step
hmc_step <- function(beta, state, X, y, eps, L) {
  draw <- nv_rnorm(shape(beta), state)
  p <- draw$values
  prop <- leapfrog(beta, p, X, y, eps, L)
  H0 <- potential(beta, X, y) + nv_sum(p^2, axes = 2L) / 2
  H1 <- potential(prop$beta, X, y) + nv_sum(prop$p^2, axes = 2L) / 2
  u <- nv_runif(shape(beta)[1L], draw$state)
  accept <- log(u$values) < H0 - H1
  accept <- nv_broadcast_to(accept, shape(beta))
  list(beta = nv_ifelse(accept, prop$beta, beta), state = u$state)
}

## ---- hmc
hmc <- jit(function(beta, state, X, y, eps, L, n_draws) {
  out <- nv_scan(list(beta = beta, state = state), steps = n_draws,
    body = function(s, x) {
      s <- hmc_step(s$beta, s$state, X, y, eps, L)
      list(carry = s, out = s$beta)
    }
  )
  list(draws = out$out, beta = out$carry$beta, state = out$carry$state)
}, static = c("L", "n_draws"))
