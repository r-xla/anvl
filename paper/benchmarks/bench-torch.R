# Usage: Rscript bench-torch.R <label> <out.csv> <experiment> [max_K]
# The device is taken from TORCH_DEVICE ("cpu" or "cuda"), the number of CPU
# threads from TORCH_THREADS.
args <- commandArgs(TRUE)
label <- args[[1L]]
out <- args[[2L]]
experiment <- args[[3L]]
max_K <- if (length(args) >= 4L) as.numeric(args[[4L]]) else Inf

suppressPackageStartupMessages(library(torch))
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
for (f in c("common.R", "grid.R", "hmc-torch.R")) source(file.path(here, f))
device <- Sys.getenv("TORCH_DEVICE", "cpu")
torch_set_num_threads(as.integer(Sys.getenv("TORCH_THREADS", "1")))
sync <- function() if (device == "cuda") cuda_synchronize()

run <- function(n, K, p, n_draws) {
  d <- sim_data(n, p)
  X <- torch_tensor(d$X, dtype = torch_float64(), device = device)
  y <- torch_tensor(as.double(d$y), dtype = torch_float64(), device = device)
  eps <- step_size(n)
  sample <- function(seed) {
    torch_manual_seed(seed)
    beta <- torch_zeros(K, p, dtype = torch_float64(), device = device)
    draws <- hmc_torch(beta, X, y, eps = eps, L = L, n_draws = n_draws)
    sync()
    draws
  }
  t_first <- system.time(draws <- sample(1L))[["elapsed"]]
  acc <- acceptance(as_array(draws$cpu()))
  times <- vapply(seq_len(n_reps), function(i) {
    system.time(sample(i + 1L))[["elapsed"]]
  }, numeric(1L))
  write_result(data.frame(
    framework = "torch", label = label, experiment = experiment, n = n, K = K,
    p = p, L = L, n_draws = n_draws, first = t_first, rep = seq_len(n_reps),
    time = times, acceptance = acc
  ), out)
}

grid <- bench_grid(experiment, max_K)
for (i in seq_len(nrow(grid))) {
  run(grid$n[[i]], grid$K[[i]], grid$p[[i]], grid$n_draws[[i]])
}
