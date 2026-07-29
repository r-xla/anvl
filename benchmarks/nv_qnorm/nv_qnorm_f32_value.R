source("f32-sweep.R")

library("anvl")

f <- function(x) ifelse(x >= 0.0 & x <= 1.0, as.double(nv_qnorm(nv_array(x, dtype="f32"))), 0.0)
g <- function(x) ifelse(x >= 0.0 & x <= 1.0, qnorm(x), 0.0)

f32_sweep(f, g, "nv_qnorm_f32_value.txt")

library(reticulate)

jax_qnorm <- function(p) {
  jnp <- import("jax.numpy", convert = FALSE)
  jstats <- import("jax.scipy.stats", convert = FALSE)
  np <- import("numpy")

  x <- jnp$asarray(p, dtype = jnp$float32)
  q <- jstats$norm$ppf(x)

  as.double(np$asarray(q))
}
f <- function(x) ifelse(x >= 0.0 & x <= 1.0, as.double(jax_qnorm(x)), 0.0)

f32_sweep(f, g, "nv_qnorm_f32_value_jax.txt")
