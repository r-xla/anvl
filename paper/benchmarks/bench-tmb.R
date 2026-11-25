# Usage: Rscript bench-tmb.R <label> <out.csv> <experiment> <cores> [max_K]
# Runs the HMC sampler of hmc-tmb.R around the TMB template logistic_tmb.cpp.
# Chains run sequentially on one core or as parallel processes on <cores> cores;
# the environment variable TMB_THREADS sets the number of OpenMP threads TMB
# uses for one chain (strong scaling). The compile time is written to
# results/overhead-tmb.csv.
args <- commandArgs(TRUE)
label <- args[[1L]]
out <- args[[2L]]
experiment <- args[[3L]]
cores <- as.integer(args[[4L]])
max_K <- if (length(args) >= 5L) as.numeric(args[[5L]]) else Inf

suppressPackageStartupMessages(library(TMB))
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
for (f in c("common.R", "grid.R", "hmc-tmb.R")) source(file.path(here, f))

build_dir <- Sys.getenv("TMB_BUILD_DIR", file.path(tempdir(), "tmb-build"))
dir.create(build_dir, showWarnings = FALSE, recursive = TRUE)
cpp <- file.path(build_dir, "logistic_tmb.cpp")
if (!file.exists(dynlib(sub("\\.cpp$", "", cpp)))) {
  file.copy(file.path(here, "logistic_tmb.cpp"), cpp, overwrite = TRUE)
  t_compile <- system.time(TMB::compile(cpp, flags = "-O2"))[["elapsed"]]
  write_result(data.frame(what = "TMB: compile", time = t_compile), file.path(dirname(out), "overhead-tmb.csv"))
}
dyn.load(dynlib(sub("\\.cpp$", "", cpp)))
TMB::openmp(as.integer(Sys.getenv("TMB_THREADS", "1")))

run <- function(n, K, p, n_draws) {
  d <- sim_data(n, p)
  obj <- MakeADFun(list(X = d$X, y = as.double(d$y)), list(beta = rep(0, p)),
    DLL = "logistic_tmb", silent = TRUE)
  eps <- step_size(n)
  sample <- function(seed) {
    one <- function(k) {
      set.seed(seed * 1000L + k)
      hmc_tmb(obj, rep(0, p), eps = eps, L = L, n_draws = n_draws)
    }
    draws <- parallel::mclapply(seq_len(K), one, mc.cores = min(cores, K))
    aperm(array(unlist(draws), c(n_draws, p, K)), c(1L, 3L, 2L))
  }
  t_first <- system.time(draws <- sample(1L))[["elapsed"]]
  acc <- acceptance(draws)
  times <- vapply(seq_len(n_reps), function(i) {
    system.time(sample(i + 1L))[["elapsed"]]
  }, numeric(1L))
  write_result(data.frame(
    framework = "tmb", label = label, experiment = experiment, n = n, K = K,
    p = p, L = L, n_draws = n_draws, first = t_first, rep = seq_len(n_reps),
    time = times, acceptance = acc
  ), out)
}

grid <- bench_grid(experiment, max_K)
for (i in seq_len(nrow(grid))) {
  run(grid$n[[i]], grid$K[[i]], grid$p[[i]], grid$n_draws[[i]])
}
