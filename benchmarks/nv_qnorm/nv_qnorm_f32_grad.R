source("f32-sweep.R")

library("anvl")

grad <- jit(gradient(\(p, log_p = FALSE) sum(nv_qnorm(p, log_p = log_p)), wrt = "p"), static = "log_p")

f <- function(x) ifelse(x >= 0.0 & x <= 1.0, as.double(grad(nv_array(x, dtype = "f32"))$p), 0.0)
g <- function(x) ifelse(x >= 0.0 & x <= 1.0, 1.0 / dnorm(qnorm(x)), 0.0)

f32_sweep(f, g, "nv_qnorm_f32_grad.txt")

library(reticulate)

jax_qnorm_grad <- function(p) {
  jax <- import("jax", convert = FALSE)
  jnp <- import("jax.numpy", convert = FALSE)
  jstats <- import("jax.scipy.stats", convert = FALSE)
  np <- import("numpy")

  x <- jnp$atleast_1d(jnp$asarray(p, dtype = jnp$float32))
  g <- jax$vmap(jax$grad(jstats$norm$ppf))(x)

  as.double(np$asarray(g))
}
f <- function(x) ifelse(x >= 0.0 & x <= 1.0, as.double(jax_qnorm_grad(x)), 0.0)

f32_sweep(f, g, "nv_qnorm_f32_grad_jax.txt")
