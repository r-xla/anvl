# Configurations of the two runtime experiments.
bench_grid <- function(experiment, max_K = Inf) {
  grid <- switch(experiment,
    single = expand.grid(n = c(1e3, 1e4, 1e5), K = 1L),
    chains = expand.grid(n = 1e4, K = 4L^(0:6)),
    smoke = expand.grid(n = c(1e3, 1e4), K = c(1L, 4L))
  )
  grid[grid$K <= max_K, , drop = FALSE]
}
