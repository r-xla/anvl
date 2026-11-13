#' ```{r setup, include = FALSE}
#' knitr::opts_chunk$set(cache = FALSE)
#' ```
# Replication script for the code shown in main.tex, extracted by extract.R.
# Render it with knitr::spin("paper_code.R"): the call in the chunk with error = TRUE
# fails on purpose, so a plain Rscript run stops there.
Sys.time()
options(width = 80)

# 3. Background: The OpenXLA compiler ecosystem

library(stablehlo)
func <- local_func("main")
x <- hlo_input("x", "f32", shape = c(2, 2))
y <- hlo_input("y", "f32", shape = c(2, 2))
func <- hlo_return(hlo_multiply(x, y))
func

# 4. Package design

# 4.1 Arrays and eager execution
library(anvl)
x <- nv_array(1:6, shape = c(2, 3), dtype = "f32")
sum(x * 2)

# 4.2 An R-like array interface
m <- matrix(1:6, nrow = 2)
cumsum(m)
as_array(cumsum(nv_array(m)))
t <- torch::torch_flatten(torch::torch_tensor(m))
torch::as_array(torch::torch_cumsum(t, dim = 1))

x <- nv_array(matrix(1:6, 2))
shape(x[1, ])
shape(x[array(1), ])

# 4.3 Tracing R code into a graph
f <- function(x, y, op) {
  if (op == "add") x + y else x * y
}
s <- nv_aval("f32", integer())
graph <- trace_fn(f, list(x = s, y = s, op = "mul"))
graph

# 4.6 The user interface
f_jit <- jit(function(x, y) {
  cat("tracing\n")
  x * y
})
y1 <- f_jit(x, x)
y2 <- f_jit(x, x + 1L)

# 5. Data types and precision

# 5.1 R values have no data type
B <- matrix(c(0.1, 0.2, 0.3, 0.4), 2, 2)
matmul <- jit(function(x, b) x %*% b)
c(dtype(matmul(nv_array(diag(2), dtype = "f32"), B)),
  dtype(matmul(nv_array(diag(2), dtype = "f64"), B)))

# 5.2 Choosing the precision
loss <- function(beta, X, y) mean((X %*% beta - y)^2)
loss_f64 <- with_dtypes(loss, c(float = "f64"))
c(dtype(nv_exp(1)), dtype(with_default_dtypes(c(float = "f64"), nv_exp(1))),
  dtype(loss_f64(nv_array(matrix(0, 2, 2), dtype = "f32"), B, B)))

# 5.3 Promotion rules
axpy <- jit(function(a, x, y) {
  args <- as_anvl_arrays(a = a, x = x, y = y, .promote = promotion_like("y"))
  args$a * args$x + args$y
})
y32 <- nv_array(c(1, 1), dtype = "f32")
as_array(axpy(2, nv_array(c(1, 2), dtype = "f32"), y32))
#+ error = TRUE
axpy(2, nv_array(c(1, 2), dtype = "f64"), y32)
#+

# 6. Extending the package

prim_softplus <- new_primitive("softplus", function(x) {
  infer <- function(x) {
    if (!(dtype(x) %in% c("f32", "f64"))) stop("`x` must be a float.")
    list(AbstractArray(dtype = dtype(x), shape = shape(x)))
  }
  graph_desc_add(self, list(x = x), infer_fn = infer)[[1L]]
})

prim_softplus[["stablehlo"]] <- function(x) {
  list(hlo_log_plus_one(hlo_exponential(x)))
}
prim_softplus[["reverse"]] <- rule_reverse(
  function(inputs, outputs, grads, params, required) {
    list(prim_mul(grads[[1L]], prim_plogis(inputs[[1L]])))
  }
)

grad_softplus <- jit(gradient(function(z) sum(prim_softplus(z))))
as_array(grad_softplus(nv_array(c(-1, 0, 1)))$z)

# 7. Use case: Hamiltonian Monte Carlo

potential <- function(beta, X, y) {
  eta <- X %*% t(beta)
  y <- nv_broadcast_to(y, shape(eta))
  loglik <- nv_sum(y * eta - nv_log1p(exp(eta)), axes = 1L)
  logprior <- -nv_sum(beta^2, axes = 2L) / 2
  -(loglik + logprior)
}

grad_potential <- gradient(
  function(beta, X, y) sum(potential(beta, X, y)),
  wrt = "beta"
)

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

hmc <- jit(function(beta, state, X, y, eps, L, n_draws) {
  out <- nv_scan(list(beta = beta, state = state), steps = n_draws,
    body = function(s, x) {
      s <- hmc_step(s$beta, s$state, X, y, eps, L)
      list(carry = s, out = s$beta)
    }
  )
  list(draws = out$out, beta = out$carry$beta, state = out$carry$state)
}, static = c("L", "n_draws"))

options(anvl.default_dtypes = c(float = "f64", int = "i32"))
set.seed(1)
n <- 1000; p <- 5
X <- matrix(rnorm(n * p), n, p)
y <- rbinom(n, 1, plogis(X %*% rnorm(p, sd = 0.5)))
res <- hmc(nv_array(matrix(0, 4, p)), nv_rng_state(42L), nv_array(X),
  nv_array(as.double(y)), eps = 0.5 / sqrt(n), L = 10L, n_draws = 2000L)
draws <- as_array(res$draws)
rbind(hmc = colMeans(apply(draws[1001:2000, , ], 3, c)),
  glm = coef(glm(y ~ X - 1, family = binomial)))

Sys.time()
sessionInfo()
