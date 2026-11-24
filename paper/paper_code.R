#' ```{r setup, include = FALSE}
#' knitr::opts_chunk$set(cache = FALSE)
#' ```
# Replication script for the code shown in main.tex, extracted by extract.R.
# Render it with knitr::spin("paper_code.R"): the call in the chunk with error = TRUE
# fails on purpose, so a plain Rscript run stops there.
Sys.time()
options(width = 80)

# 3. Using anvl

# 3.2 Array construction and operations
library(anvl)
nv_set_seed(1L)
x <- nv_array(1:6, shape = c(2, 3), dtype = "f32")
sum(x * 2)
nv_rnorm(3L)

as_array(x)

# 3.3 Compilation
f <- function(x) sum(x^2)
f_jit <- jit(f)
f_jit(x)

# 3.4 Differentiation
g <- function(x, w) sum(w * x^2)
grad_g <- jit(gradient(g, wrt = "x"))
grad_g(x, 2)$x

# 3.5 Choosing the precision and device
c(dtype(nv_scalar(pi)), dtype(nv_exp(1)))
options(anvl.default_dtypes = c(float = "f64", int = "i32"))
c(dtype(nv_scalar(pi)), dtype(nv_exp(1)), dtype(x * pi))

# 3.6 Hamiltonian Monte Carlo
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

hmc_step <- function(beta, X, y, eps, L) {
  p <- nv_rnorm(shape(beta))
  prop <- leapfrog(beta, p, X, y, eps, L)
  H0 <- potential(beta, X, y) + nv_sum(p^2, axes = 2L) / 2
  H1 <- potential(prop$beta, X, y) + nv_sum(prop$p^2, axes = 2L) / 2
  u <- nv_runif(shape(beta)[1L])
  accept <- log(u) < H0 - H1
  accept <- nv_broadcast_to(accept, shape(beta))
  nv_ifelse(accept, prop$beta, beta)
}

hmc_step_jit <- jit(hmc_step, static = "L")
hmc <- function(beta, X, y, eps, L, n_draws) {
  draws <- vector("list", n_draws)
  for (i in seq_len(n_draws)) {
    beta <- hmc_step_jit(beta, X, y, eps, L)
    draws[[i]] <- beta
  }
  simplify2array(lapply(draws, as_array))
}

n <- 1000; p <- 2
X <- nv_rnorm(c(n, p))
eta <- X %*% nv_rnorm(c(p, 1L), sd = 0.5)
y <- nv_ifelse(nv_runif(c(n, 1L)) < nv_plogis(eta), 1, 0)
draws <- hmc(nv_array(matrix(0, 4, p)), X, y, eps = 0.5 / sqrt(n),
  L = 20L, n_draws = 2000L)
fit <- glm(as_array(y) ~ as_array(X) - 1, family = binomial)
rbind(hmc = apply(draws[, , 1001:2000], 2, mean), glm = unname(coef(fit)))

# 4. How anvl works

f <- function(x, y) x * y
s <- nv_aval("f32", integer())
graph <- trace_fn(f, list(x = s, y = s))
graph

grad_graph <- graph_gradient(graph, wrt = "x")
grad_graph

stablehlo(grad_graph)[[1L]]

# 4.1 Tracing R code into an AnvlGraph
g <- function(x, y, op) if (op == "add") x + y else x * y
trace_fn(g, list(x = s, y = s, op = "mul"))

h <- function(x, y) nv_if(x > y, function() x, function() y)
trace_fn(h, list(x = s, y = s))

times2 <- function(x, n) {
  for (i in seq_len(n)) x <- x * 2
  x
}
trace_fn(times2, list(x = s, n = 3L))

# 4.3 Lowering and compilation
with_backend("quickr", {
  z <- nv_array(c(1, 2, 3))
  jit(function(z) z * 2)(z)
})

library(stablehlo)
func <- local_func("main")
x <- hlo_input("x", "f32", shape = c(2, 2))
y <- hlo_input("y", "f32", shape = c(2, 2))
func <- hlo_return(hlo_multiply(x, y))
func

dev = "cpu"
library(pjrt)
exec <- pjrt_compile(pjrt_program(repr(func)), device = dev)
a <- pjrt_buffer(matrix(1:4, 2), dtype = "f32", device = dev)
b <- pjrt_buffer(matrix(5:8, 2), dtype = "f32", device = dev)
pjrt_execute(exec, a, b)

# 4.4 The lazy user interface
f_jit <- jit(function(x, y) {
  cat("tracing\n")
  x * y
})
v <- nv_array(1:3)
y1 <- f_jit(v, v)
y2 <- f_jit(v, v + 1L)

# 4.7 An R-like array interface
m <- matrix(1:6, nrow = 2)
cumsum(m)
cumsum(nv_array(m))
t <- torch::torch_flatten(torch::torch_tensor(m))
torch::torch_cumsum(t, dim = 1)

x <- nv_array(matrix(1:6, 2))
shape(x[1, ])
shape(x[array(1), ])

# 4.8 Type system
B <- matrix(c(0.1, 0.2, 0.3, 0.4), 2, 2)
matmul <- jit(function(x, b) x %*% b)
c(dtype(matmul(nv_array(diag(2), dtype = "f32"), B)),
  dtype(matmul(nv_array(diag(2), dtype = "f64"), B)))

axpy <- jit(function(a, x, y) {
  args <- as_anvl_arrays(a = a, x = x, y = y, .promote = promotion_like("y"))
  args$a * args$x + args$y
})
y32 <- nv_array(c(1, 1), dtype = "f32")
axpy(2, nv_array(c(1, 2), dtype = "f32"), y32)
#+ error = TRUE
axpy(2, nv_array(c(1, 2), dtype = "f64"), y32)
#+

Sys.time()
sessionInfo()
