## ---------------------------------------------------------------------------
## nv_dbinom(), nv_pbinom(), nv_qbinom() against base R
##
## Each function is timed at f32 and f64 next to its base R counterpart, with
## `-x` as a baseline: it touches every element and cannot be folded away, so
## it measures the fixed cost of a call and of bringing the result back to R.
## `as.vector()` forces that transfer -- anvl executes asynchronously, so a bare
## `f(x)` returns before the computation has finished.
##
## The inputs are written to `inputs/` for `binomial.py`, which times JAX and
## SciPy on exactly the same values. Run this script first, then that one, and
## never both at once: both XLA runtimes use every core.
##
##   Rscript benchmarks/binomial/binomial.R
##
## Sample sizes are chosen so that each of the three parts takes about 10
## seconds on an Apple M1. The quantile search loops until every element has
## converged, so its cost depends on the hardest values in the sample -- which
## is why the inputs are fixed by a seed and shared rather than drawn per run.
## ---------------------------------------------------------------------------

library(anvl)

args <- commandArgs(FALSE)
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", args, value = TRUE))))
inputs <- file.path(here, "inputs")
dir.create(inputs, showWarnings = FALSE)

size <- 100
prob <- 0.3
n <- c(d = 1.5e6, p = 5e5, q = 1.5e5)

set.seed(1)
x_d <- rbinom(n[["d"]], size, prob)
x_p <- rbinom(n[["p"]], size, prob)
p_q <- runif(n[["q"]])
writeBin(as.double(x_d), file.path(inputs, "x_d.bin"))
writeBin(as.double(x_p), file.path(inputs, "x_p.bin"))
writeBin(p_q, file.path(inputs, "p_q.bin"))

base <- jit(function(x) -x)
baseR <- function(x) -x

# f is the anvl function, fR its base R counterpart
bench_one <- function(f, fR, x) {
  x_f32 <- nv_array(x, dtype = "f32")
  x_f64 <- nv_array(x, dtype = "f64")
  # Compile + warm
  invisible(as.vector(f(x_f32)))
  invisible(as.vector(base(x_f32)))
  invisible(as.vector(f(x_f64)))
  invisible(as.vector(base(x_f64)))

  bench::mark(
    f_32 = as.vector(f(x_f32)),
    base_32 = as.vector(base(x_f32)),
    f_64 = as.vector(f(x_f64)),
    base_64 = as.vector(base(x_f64)),
    fR = as.vector(fR(x)),
    baseR = as.vector(baseR(x)),
    min_iterations = 50,
    check = FALSE,
    memory = FALSE
  )
}

parts <- list(
  dbinom = list(
    f = jit(function(x) nv_dbinom(x, size, prob)),
    fR = \(x) dbinom(x, size, prob),
    x = x_d
  ),
  pbinom = list(
    f = jit(function(q) nv_pbinom(q, size, prob)),
    fR = \(q) pbinom(q, size, prob),
    x = x_p
  ),
  qbinom = list(
    f = jit(function(p) nv_qbinom(p, size, prob)),
    fR = \(p) qbinom(p, size, prob),
    x = p_q
  )
)

for (name in names(parts)) {
  part <- parts[[name]]
  start <- proc.time()[["elapsed"]]
  b <- bench_one(part$f, part$fR, part$x)
  elapsed <- proc.time()[["elapsed"]] - start
  cat(sprintf("\n== %s: n = %g, size = %g, prob = %g (%.1f s)\n", name, length(part$x), size, prob, elapsed))
  print(b[, c("expression", "min", "median", "n_itr", "n_gc")])
}
