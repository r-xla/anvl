# Shared settings of the runtime benchmark.
p <- 20L
L <- 10L
n_draws <- 1000L
n_reps <- 3L

sim_data <- function(n, p, seed = 1L) {
  withr::with_seed(seed, {
    X <- matrix(rnorm(n * p), n, p)
    beta <- rnorm(p, sd = 0.5)
    y <- rbinom(n, 1L, plogis(drop(X %*% beta)))
  })
  list(X = X, y = y)
}

# Step size that keeps the acceptance rate high as the posterior contracts.
step_size <- function(n) 0.5 / sqrt(n)

# Fraction of draws (after the first) in which a chain moved.
acceptance <- function(draws) {
  # draws: (n_draws, K, p)
  moved <- apply(draws, 2L, function(d) rowSums(abs(diff(d))) > 0)
  mean(moved)
}

write_result <- function(row, file) {
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  utils::write.table(row, file, sep = ",", row.names = FALSE,
    col.names = !file.exists(file), append = file.exists(file))
}
