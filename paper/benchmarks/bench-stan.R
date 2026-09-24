# Usage: Rscript bench-stan.R compile
#        Rscript bench-stan.R <label> <out.csv> <experiment> <cores> [max_K]
#        Rscript bench-stan.R <label> <out.csv> <experiment> <threads> threads
#        Rscript bench-stan.R <label> <out.csv> <experiment> <cores> opencl [max_K]
# With "threads", one chain runs the model of logistic-threads.stan, whose
# likelihood reduce_sum() parallelizes over <threads> threads. With "opencl",
# logistic.stan is compiled with STAN_OPENCL, which evaluates
# bernoulli_logit_glm() on the first OpenCL device (the GPU).
# Runs CmdStan's static HMC with the same step size, number of leapfrog steps,
# and unit metric as the anvl sampler. Static HMC supports one chain per
# process, so K chains are K processes, of which <cores> run at the same time.
# The sampling time is the wall time of a run minus that of an otherwise
# identical run without draws, which removes process start-up and data parsing.
args <- commandArgs(TRUE)

suppressPackageStartupMessages(library(cmdstanr))
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
for (f in c("common.R", "grid.R")) source(file.path(here, f))

build_dir <- Sys.getenv("STAN_BUILD_DIR", file.path(tempdir(), "stan-build"))
dir.create(build_dir, showWarnings = FALSE, recursive = TRUE)
stan_file <- file.path(build_dir, "logistic.stan")
if (!file.exists(stan_file) || !identical(readLines(stan_file), readLines(file.path(here, "logistic.stan")))) {
  file.copy(file.path(here, "logistic.stan"), stan_file, overwrite = TRUE)
}
model <- cmdstan_model(stan_file, quiet = TRUE)
threads_file <- file.path(build_dir, "logistic-threads.stan")
file.copy(file.path(here, "logistic-threads.stan"), threads_file, overwrite = TRUE)
model_threads <- cmdstan_model(threads_file, cpp_options = list(stan_threads = TRUE), quiet = TRUE)
opencl_file <- file.path(build_dir, "logistic-opencl.stan")
file.copy(file.path(here, "logistic.stan"), opencl_file, overwrite = TRUE)
compile_opencl <- function() {
  cmdstan_model(opencl_file, cpp_options = list(stan_opencl = TRUE), quiet = TRUE)
}
if (identical(args[[1L]], "compile")) {
  if (identical(Sys.getenv("STAN_COMPILE_OPENCL"), "1")) compile_opencl()
  quit(save = "no")
}

label <- args[[1L]]
out <- args[[2L]]
experiment <- args[[3L]]
cores <- as.integer(args[[4L]])
use_threads <- length(args) >= 5L && identical(args[[5L]], "threads")
use_opencl <- length(args) >= 5L && identical(args[[5L]], "opencl")
max_K <- if (use_opencl && length(args) >= 6L) {
  as.numeric(args[[6L]])
} else if (length(args) >= 5L && !use_threads && !use_opencl) {
  as.numeric(args[[5L]])
} else {
  Inf
}
if (use_opencl) model_opencl <- compile_opencl()
work <- tempfile("stan")
dir.create(work)

stan_run <- function(data_file, n, K, draws, seed) {
  exe <- if (use_threads) {
    model_threads$exe_file()
  } else if (use_opencl) {
    model_opencl$exe_file()
  } else {
    model$exe_file()
  }
  one <- function(k) {
    system2(exe, c("sample", sprintf("num_samples=%d", draws),
      "num_warmup=0", "adapt", "engaged=0", "algorithm=hmc", "engine=static",
      sprintf("int_time=%.17g", L * step_size(n)),
      sprintf("stepsize=%.17g", step_size(n)), "metric=unit_e", "init=0",
      sprintf("id=%d", k), "data", sprintf("file=%s", data_file),
      "output", sprintf("file=%s", file.path(work, sprintf("out_%d.csv", k))),
      "random", sprintf("seed=%d", seed), if (use_threads) sprintf("num_threads=%d", cores),
      if (use_opencl) c("opencl", "platform=0", "device=0")),
      stdout = FALSE)
  }
  processes <- if (use_threads) 1L else min(cores, K)
  system.time(parallel::mclapply(seq_len(K), one, mc.cores = processes))[["elapsed"]]
}

read_draws <- function(K, p) {
  draws <- lapply(seq_len(K), function(k) {
    f <- file.path(work, sprintf("out_%d.csv", k))
    as.matrix(utils::read.csv(f, comment.char = "#")[, paste0("beta.", seq_len(p))])
  })
  aperm(array(unlist(draws), c(nrow(draws[[1L]]), p, K)), c(1L, 3L, 2L))
}

run <- function(n, K, p, n_draws) {
  d <- sim_data(n, p)
  data_file <- file.path(work, "data.json")
  write_stan_json(list(n = n, p = p, X = d$X, y = d$y, grainsize = 1L), data_file)
  full <- stan_run(data_file, n, K, n_draws, 1L)
  acc <- acceptance(read_draws(K, p))
  base <- stan_run(data_file, n, K, 0L, 1L)
  times <- vapply(seq_len(n_reps), function(i) {
    stan_run(data_file, n, K, n_draws, i + 1L) - stan_run(data_file, n, K, 0L, i + 1L)
  }, numeric(1L))
  write_result(data.frame(
    framework = "stan", label = label, experiment = experiment, n = n, K = K,
    p = p, L = L, n_draws = n_draws, first = full - base,
    rep = seq_len(n_reps), time = times, acceptance = acc
  ), out)
}

grid <- bench_grid(experiment, max_K)
for (i in seq_len(nrow(grid))) {
  run(grid$n[[i]], grid$K[[i]], grid$p[[i]], grid$n_draws[[i]])
}
