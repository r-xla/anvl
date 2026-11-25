# Configurations of the runtime experiments: number of observations n, number of
# chains K, number of features p, and number of draws per chain. The environment
# variable BENCH_MIN_K skips configurations with fewer chains, to resume a run.
bench_grid <- function(experiment, max_K = Inf) {
  grid <- switch(experiment,
    single = data.frame(n = c(1e3, 1e4, 1e5), K = 1L, p = 20L, n_draws = 1000L),
    chains = data.frame(n = 1e4, K = 4L^(0:6), p = 20L, n_draws = 1000L),
    scaling = data.frame(n = 1e5, K = 1L, p = 100L, n_draws = 200L),
    scaling_chains = data.frame(n = 1e4, K = 64L, p = 20L, n_draws = 200L),
    smoke = data.frame(n = c(1e3, 1e4), K = c(1L, 4L), p = 20L, n_draws = 100L)
  )
  min_K <- as.numeric(Sys.getenv("BENCH_MIN_K", "0"))
  grid[grid$K <= max_K & grid$K >= min_K, , drop = FALSE]
}
