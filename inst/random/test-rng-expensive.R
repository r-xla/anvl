# Expensive RNG tests for statistical verification
# They are slow, so we don't run them in the normal CI suite

library(anvl)

test_rnorm_statistical <- function() {
  cat("Testing rnorm statistical properties...\n")

  # Test normality with large sample
  f <- function() {
    nv_rnorm(nv_array(c(1, 2), dtype = "ui64"), dtype = "f64", shape = c(200L, 300L, 400L))
  }
  g <- jit(f)
  out <- g()
  Z <- as_array(out[[2]])

  # Check no non-finite values
  stopifnot(all(is.finite(Z)))

  # Shapiro-Wilk test on slices

  tst <- apply(Z, c(2, 3), \(z) shapiro.test(z)$p.value)
  rejection_rate <- mean(c(tst) < 0.05)
  cat(sprintf("  Shapiro-Wilk rejection rate at alpha=0.05: %.3f (expected ~0.05)\n", rejection_rate))
  stopifnot(abs(rejection_rate - 0.05) < 0.02)

  cat("  PASS\n")
}

test_rnorm_mean_sd <- function() {
  cat("Testing rnorm mean and standard deviation...\n")

  f <- function() {
    nv_rnorm(
      nv_array(c(3, 83), dtype = "ui64"),
      dtype = "f64",
      shape = c(10L, 10L, 10L, 10L, 10L),
      mean = 10,
      sd = 9
    )
  }
  g <- jit(f)
  out <- g()
  values <- as_array(out[[2]])

  sample_mean <- mean(values)
  sample_sd <- sd(values)

  cat(sprintf("  Sample mean: %.2f (expected 10)\n", sample_mean))
  cat(sprintf("  Sample SD: %.2f (expected 9)\n", sample_sd))

  stopifnot(abs(sample_mean - 10) < 0.5)
  stopifnot(abs(sample_sd - 9) < 0.5)

  cat("  PASS\n")
}

test_runif_statistical <- function() {
  cat("Testing runif statistical properties...\n")

  f <- function() {
    nv_runif(
      nv_array(c(1, 2), dtype = "ui64"),
      dtype = "f32",
      shape = c(10, 20, 30, 40, 50),
      min = -1,
      max = 1
    )
  }
  g <- jit(f)
  out <- g()
  values <- as_array(out[[2]])

  # Check bounds (should be open interval)
  stopifnot(!any(values == -1))
  stopifnot(!any(values == 1))

  # Check mean (expected 0 for U[-1, 1])
  sample_mean <- mean(values)
  cat(sprintf("  Sample mean: %.5f (expected 0)\n", sample_mean))
  stopifnot(abs(sample_mean) < 1e-3)

  # Check variance (expected 1/3 for U[-1, 1])
  sample_var <- var(values)
  stopifnot(abs(sample_var - 1 / 3) < 1e-3)

  cat("  PASS\n")
}

test_rbinom_statistical <- function() {
  cat("Testing nv_rbinom statistical properties...\n")

  # Test Bernoulli case (size = 1)
  cat("  Testing Bernoulli (n=1)...\n")
  f <- function() {
    nv_rbinom(
      nv_array(c(1, 2), dtype = "ui64"),
      dtype = "i32",
      shape = c(100L, 100L, 100L)
    )
  }
  g <- jit(f)
  out <- g()
  values <- as_array(out[[2]])

  # All values should be 0 or 1
  stopifnot(all(values %in% c(0L, 1L)))

  prop_ones <- mean(values)
  cat(sprintf("    Proportion of 1s: %.5f (expected 0.5)\n", prop_ones))
  stopifnot(abs(prop_ones - 0.5) < 0.005)

  sample_var <- var(c(values))
  cat(sprintf("    Sample variance: %.5f (expected 0.25)\n", sample_var))
  stopifnot(abs(sample_var - 0.25) < 0.005)

  # Test Binomial case (size = 20, prob = 0.5)
  # For Binomial(n=20, p=0.5): mean = 10, variance = 5
  cat("  Testing Binomial (n=20, prob=0.5)...\n")
  f2 <- function() {
    nv_rbinom(
      nv_array(c(3, 4), dtype = "ui64"),
      size = 20L,
      dtype = "i32",
      shape = c(100L, 100L, 100L)
    )
  }
  g2 <- jit(f2)
  out2 <- g2()
  values2 <- as_array(out2[[2]])

  # All values should be in [0, 20]
  stopifnot(all(values2 >= 0L & values2 <= 20L))

  sample_mean <- mean(values2)
  cat(sprintf("    Sample mean: %.4f (expected 10)\n", sample_mean))
  stopifnot(abs(sample_mean - 10) < 0.05)

  sample_var <- var(c(values2))
  cat(sprintf("    Sample variance: %.4f (expected 5)\n", sample_var))
  stopifnot(abs(sample_var - 5) < 0.05)

  # Test Binomial case with prob = 0.3
  # For Binomial(n=10, p=0.3): mean = 3, variance = 2.1
  cat("  Testing Binomial (n=10, prob=0.3)...\n")
  f3 <- function() {
    nv_rbinom(
      nv_array(c(5, 6), dtype = "ui64"),
      size = 10L,
      prob = 0.3,
      dtype = "i32",
      shape = c(100L, 100L, 100L)
    )
  }
  g3 <- jit(f3)
  out3 <- g3()
  values3 <- as_array(out3[[2]])

  # All values should be in [0, 10]
  stopifnot(all(values3 >= 0L & values3 <= 10L))

  sample_mean3 <- mean(values3)
  cat(sprintf("    Sample mean: %.4f (expected 3)\n", sample_mean3))
  stopifnot(abs(sample_mean3 - 3) < 0.05)

  sample_var3 <- var(c(values3))
  cat(sprintf("    Sample variance: %.4f (expected 2.1)\n", sample_var3))
  stopifnot(abs(sample_var3 - 2.1) < 0.05)

  # Test Bernoulli with prob = 0.7
  # For Bernoulli(p=0.7): mean = 0.7, variance = 0.21
  cat("  Testing Bernoulli (prob=0.7)...\n")
  f4 <- function() {
    nv_rbinom(
      nv_array(c(7, 8), dtype = "ui64"),
      prob = 0.7,
      dtype = "i32",
      shape = c(100L, 100L, 100L)
    )
  }
  g4 <- jit(f4)
  out4 <- g4()
  values4 <- as_array(out4[[2]])

  # All values should be 0 or 1
  stopifnot(all(values4 %in% c(0L, 1L)))

  sample_mean4 <- mean(values4)
  cat(sprintf("    Sample mean: %.4f (expected 0.7)\n", sample_mean4))
  stopifnot(abs(sample_mean4 - 0.7) < 0.01)

  sample_var4 <- var(c(values4))
  cat(sprintf("    Sample variance: %.4f (expected 0.21)\n", sample_var4))
  stopifnot(abs(sample_var4 - 0.21) < 0.01)

  cat("  PASS\n")
}

test_sample_statistical <- function() {
  cat("Testing nv_sample statistical properties...\n")

  cat("  Testing equal probabilities...\n")
  f1 <- function() {
    nv_sample(
      n = 6L,
      shape = 60000L,
      initial_state = nv_array(c(1, 2), dtype = "ui64")
    )
  }
  g1 <- jit(f1)
  out1 <- g1()
  values1 <- as_array(out1[[2]])

  stopifnot(all(values1 >= 1L & values1 <= 6L))

  for (i in 1:6) {
    prop <- mean(values1 == i)
    cat(sprintf("    Category %d: %.4f (expected ~0.1667)\n", i, prop))
    stopifnot(abs(prop - 1 / 6) < 0.01)
  }

  cat("  PASS\n")
}

run_all_tests <- function() {
  cat("Running expensive RNG tests...\n\n")

  test_rnorm_statistical()
  test_rnorm_mean_sd()
  test_runif_statistical()
  test_rbinom_statistical()
  test_sample_statistical()

  cat("\nAll expensive RNG tests passed!\n")
}

run_all_tests()
