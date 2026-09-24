# Usage: Rscript bench-anvl.R <label> <out.csv> <experiment> [max_K]
# The device is taken from ANVL_DEFAULT_DEVICE, the number of CPU cores from
# the affinity mask the process runs with.
args <- commandArgs(TRUE)
label <- args[[1L]]
out <- args[[2L]]
experiment <- args[[3L]]
max_K <- if (length(args) >= 4L) as.numeric(args[[4L]]) else Inf

suppressPackageStartupMessages(library(anvl))
options(anvl.default_dtypes = c(float = "f64", int = "i32"))
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
for (f in c("common.R", "grid.R", "hmc-anvl.R")) source(file.path(here, f))

run <- function(n, K) {
  d <- sim_data(n, p)
  X <- nv_array(d$X)
  y <- nv_array(as.double(d$y))
  eps <- step_size(n)
  sample <- function(seed) {
    res <- hmc(nv_array(matrix(0, K, p)), nv_rng_state(seed), X, y,
      eps = eps, L = L, n_draws = n_draws)
    await(res$draws)
    res
  }
  t_first <- system.time(res <- sample(1L))[["elapsed"]]
  acc <- acceptance(as_array(res$draws))
  times <- vapply(seq_len(n_reps), function(i) {
    system.time(sample(i + 1L))[["elapsed"]]
  }, numeric(1L))
  write_result(data.frame(
    framework = "anvl", label = label, experiment = experiment, n = n, K = K,
    p = p, L = L, n_draws = n_draws, first = t_first, rep = seq_len(n_reps),
    time = times, acceptance = acc
  ), out)
}

grid <- bench_grid(experiment, max_K)
for (i in seq_len(nrow(grid))) {
  run(grid$n[[i]], grid$K[[i]])
}
