# Usage: Rscript overhead.R <out.csv>
# Splits the one-off cost of the first call into tracing, lowering to StableHLO
# and XLA compilation, and compares the compile times of the pjrt and quickr
# backends (for the gradient, which quickr can compile) and of CmdStan.
args <- commandArgs(TRUE)
out <- args[[1L]]

suppressPackageStartupMessages({
  library(anvl)
  library(cmdstanr)
})
options(anvl.default_dtypes = c(float = "f64", int = "i32"))
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "common.R"))
source(file.path(here, "hmc-anvl.R"))

n <- 1e4
p <- 20L
n_draws <- 1000L
d <- sim_data(n, p)
X <- nv_array(d$X)
y <- nv_array(as.double(d$y))
beta <- nv_array(matrix(0, 1L, p))
elapsed <- function(expr) system.time(expr)[["elapsed"]]
rows <- list()
add <- function(what, time) rows[[length(rows) + 1L]] <<- data.frame(what = what, time = time)

# Phases of the first call of the sampler.
f <- function(beta, state, X, y) hmc(beta, state, X, y, eps = step_size(n), L = L, n_draws = n_draws)
args <- list(beta = beta, state = nv_rng_state(1L), X = X, y = y)
add("hmc: trace", elapsed(graph <- trace_fn(f, args)))
add("hmc: lower to StableHLO", elapsed(src <- stablehlo::repr(stablehlo(graph)[[1L]])))
add("hmc: XLA compile", elapsed(pjrt::pjrt_compile(pjrt::pjrt_program(src = src, format = "mlir"))))
f_jit <- jit(f)
add("hmc: first call", elapsed(await(f_jit(beta, nv_rng_state(1L), X, y)$draws)))
add("hmc: cached call", elapsed(await(f_jit(beta, nv_rng_state(2L), X, y)$draws)))

# The gradient alone, on both backends.
g <- function(beta, X, y) grad_potential(beta, X, y)$beta
g_pjrt <- jit(g)
add("gradient (pjrt): first call", elapsed(await(g_pjrt(beta, X, y))))
add("gradient (pjrt): cached call", median(vapply(1:20, function(i) elapsed(await(g_pjrt(beta, X, y))), numeric(1L))))
with_backend("quickr", {
  Xq <- nv_array(d$X)
  yq <- nv_array(as.double(d$y))
  bq <- nv_array(matrix(0, 1L, p))
  g_quickr <- jit(g)
  add("gradient (quickr): first call", elapsed(g_quickr(bq, Xq, yq)))
  add("gradient (quickr): cached call", median(vapply(1:20, function(i) elapsed(g_quickr(bq, Xq, yq)), numeric(1L))))
})

# CmdStan: translation to C++ and compilation.
work <- tempfile("stan")
dir.create(work)
file.copy(file.path(here, "logistic.stan"), work)
add("CmdStan: compile", elapsed(cmdstan_model(file.path(work, "logistic.stan"), force_recompile = TRUE, quiet = TRUE)))

res <- do.call(rbind, rows)
print(res)
write_result(res, out)
