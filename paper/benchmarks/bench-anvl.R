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

# With PJRT_CPU_DEVICE_COUNT > 1, the chains are split into one group per device
# and each group runs as a separate program; like CmdStan's one process per chain,
# this lets every core work on its own chains.
n_devices <- as.integer(Sys.getenv("PJRT_CPU_DEVICE_COUNT", "1"))

run <- function(n, K, p, n_draws) {
  d <- sim_data(n, p)
  eps <- step_size(n)
  G <- min(K, n_devices)
  devices <- if (G > 1L) lapply(seq_len(G) - 1L, function(i) nv_device(sprintf("cpu:%d", i)))
  sizes <- diff(round(seq(0, K, length.out = G + 1L)))
  data <- lapply(seq_len(G), function(g) {
    list(X = nv_array(d$X, device = devices[[g]]), y = nv_array(as.double(d$y), device = devices[[g]]))
  })
  sample <- function(seed) {
    res <- lapply(seq_len(G), function(g) {
      hmc(nv_array(matrix(0, sizes[[g]], p), device = devices[[g]]),
        nv_rng_state(seed * 1000L + g, device = devices[[g]]), data[[g]]$X, data[[g]]$y,
        eps = eps, L = L, n_draws = n_draws)
    })
    lapply(res, function(r) await(r$draws))
  }
  t_first <- system.time(res <- sample(1L))[["elapsed"]]
  draws <- lapply(res, as_array)
  acc <- sum(vapply(seq_len(G), function(g) acceptance(draws[[g]]) * sizes[[g]], numeric(1L))) / K
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
  run(grid$n[[i]], grid$K[[i]], grid$p[[i]], grid$n_draws[[i]])
}
