# The HMC sampler of hmc-anvl.R, written with torch: the gradient comes from
# torch's autograd, and the loops are R loops, as torch executes eagerly.

potential_torch <- function(beta, X, y) {
  eta <- torch_matmul(X, beta$t())
  loglik <- (y$unsqueeze(2L) * eta - torch_log1p(torch_exp(eta)))$sum(dim = 1L)
  logprior <- -(beta^2)$sum(dim = 2L) / 2
  -(loglik + logprior)
}

grad_potential_torch <- function(beta, X, y) {
  with_enable_grad({
    beta <- beta$detach()$requires_grad_(TRUE)
    autograd_grad(potential_torch(beta, X, y)$sum(), beta)[[1L]]
  })
}

hmc_torch <- function(beta, X, y, eps, L, n_draws) {
  with_no_grad({
    draws <- torch_empty(c(n_draws, dim(beta)), dtype = beta$dtype, device = beta$device)
    for (i in seq_len(n_draws)) {
      rho <- torch_randn_like(beta)
      b <- beta
      r <- rho
      grad <- grad_potential_torch(b, X, y)
      for (l in seq_len(L)) {
        r <- r - eps / 2 * grad
        b <- b + eps * r
        grad <- grad_potential_torch(b, X, y)
        r <- r - eps / 2 * grad
      }
      H0 <- potential_torch(beta, X, y) + (rho^2)$sum(dim = 2L) / 2
      H1 <- potential_torch(b, X, y) + (r^2)$sum(dim = 2L) / 2
      accept <- torch_log(torch_rand(dim(beta)[1L], dtype = beta$dtype, device = beta$device)) < H0 - H1
      beta <- torch_where(accept$unsqueeze(2L), b, beta)
      draws[i, , ] <- beta
    }
    draws
  })
}
