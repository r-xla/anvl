test_that("nv_rnorm", {
  # statistical validity checks are in inst/random
  out <- nv_rnorm(nv_array(c(1, 2), dtype = "ui64"), dtype = "f32", shape = c(2, 3))
  expect_equal(dtype(out[[1]]), as_dtype("ui64"))
  expect_equal(shape(out[[1]]), 2L)
  expect_equal(dtype(out[[2]]), as_dtype("f32"))
  expect_equal(shape(out[[2]]), c(2L, 3L))

  # test with uneven total number of RVs
  out <- nv_rnorm(nv_array(c(1, 2), dtype = "ui64"), dtype = "f32", shape = c(3, 3))
  expect_equal(shape(out[[2]]), c(3L, 3L))

  # test mu/sigma parameters with small sample
  out <- nv_rnorm(
    nv_array(c(3, 83), dtype = "ui64"),
    dtype = "f64",
    shape = c(2L, 3L),
    mu = 10,
    sigma = 9
  )
  expect_equal(dtype(out[[1]]), as_dtype("ui64"))
  expect_equal(shape(out[[1]]), 2L)
  expect_equal(shape(out[[2]]), c(2L, 3L))
  expect_equal(dtype(out[[2]]), as_dtype("f64"))
})

test_that("rng rejects non-f32/f64 dtypes", {
  key <- nv_array(c(1, 2), dtype = "ui64")
  expect_error(
    nv_rnorm(key, dtype = "bf16", shape = 2L),
    "must be a floating-point dtype \\(f32 or f64\\)"
  )
  expect_error(
    nv_rnorm(key, dtype = "i32", shape = 2L),
    "must be a floating-point dtype \\(f32 or f64\\)"
  )
})

test_that("nv_runif", {
  # statistical validity checks are in inst/random
  out <- nv_runif(
    nv_array(c(1, 2), dtype = "ui64"),
    dtype = "f32",
    shape = c(3, 4),
    lower = -1,
    upper = 1
  )

  expect_equal(dtype(out[[1]]), as_dtype("ui64"))
  expect_equal(shape(out[[1]]), 2L)
  expect_equal(shape(out[[2]]), c(3L, 4L))
  expect_equal(dtype(out[[2]]), as_dtype("f32"))
})

test_that("nv_rbinom", {
  # statistical validity checks are in inst/random
  out <- nv_rbinom(nv_array(c(1, 2), dtype = "ui64"), dtype = "i32", shape = c(2, 5))

  expect_equal(shape(out[[1]]), 2L)
  expect_equal(shape(out[[2]]), c(2L, 5L))
  expect_equal(dtype(out[[2]]), as_dtype("i32"))

  # All values should be 0 or 1
  values <- c(as_array(out[[2]]))
  expect_true(all(values %in% c(0L, 1L)))

  # Test with different dtype
  out2 <- nv_rbinom(nv_array(c(1, 2), dtype = "ui64"), dtype = "f32", shape = 10L)
  expect_equal(dtype(out2[[2]]), as_dtype("f32"))
  expect_equal(shape(out2[[2]]), 10L)

  # Test with non-multiple-of-8 shape (tests slicing)
  out3 <- nv_rbinom(nv_array(c(1, 2), dtype = "ui64"), dtype = "i32", shape = c(3, 3))
  expect_equal(shape(out3[[2]]), c(3L, 3L))
})

test_that("nv_sample with replacement", {
  # statistical validity checks are in inst/random
  state <- nv_array(c(1, 2), dtype = "ui64")

  # Test with equal probabilities
  out1 <- nv_sample(x = 6L, shape = 10L, initial_state = state, replace = TRUE)

  expect_equal(shape(out1[[1]]), 2L)
  expect_equal(shape(out1[[2]]), 10L)
  expect_equal(dtype(out1[[2]]), as_dtype("i32"))

  # All values should be in 1:6
  values1 <- c(as_array(out1[[2]]))
  expect_true(all(values1 >= 1L & values1 <= 6L))

  # Test 2D output shape
  out3 <- nv_sample(x = 4L, shape = c(2L, 3L), initial_state = state, replace = TRUE)
  expect_equal(shape(out3[[2]]), c(2L, 3L))

  # The last category is reachable and the first is not over-represented
  values4 <- c(as_array(nv_sample(x = 6L, shape = 5000L, initial_state = state, replace = TRUE)[[2]]))
  expect_setequal(unique(values4), 1:6)
  expect_true(all(abs(as.numeric(table(values4)) / 5000 - 1 / 6) < 0.02))

  # Zero-weight categories are never drawn
  values5 <- c(as_array(nv_sample(
    x = 3L,
    shape = 200L,
    initial_state = state,
    replace = TRUE,
    probs = nv_array(c(1, 0, 1))
  )[[2]]))
  expect_setequal(unique(values5), c(1L, 3L))
})

test_that("nv_sample without replacement", {
  state <- nv_array(c(1, 2), dtype = "ui64")

  # Drawing the whole population yields a permutation of it
  perm <- c(as_array(nv_sample(x = 6L, shape = 6L, initial_state = state)[[2]]))
  expect_setequal(perm, 1:6)

  # A partial draw has no duplicates
  part <- c(as_array(nv_sample(x = 10L, shape = 4L, initial_state = state)[[2]]))
  expect_length(unique(part), 4L)
  expect_true(all(part >= 1L & part <= 10L))

  # 2D output shape
  expect_equal(shape(nv_sample(x = 9L, shape = c(3L, 3L), initial_state = state)[[2]]), c(3L, 3L))

  # Cannot draw more than the population
  expect_error(
    nv_sample(x = 6L, shape = 10L, initial_state = state),
    "Cannot take a sample larger than the population"
  )

  # Zero-weight categories are never drawn
  drawn <- c(as_array(nv_sample(
    x = 3L,
    shape = 2L,
    initial_state = state,
    probs = nv_array(c(1, 0, 1))
  )[[2]]))
  expect_setequal(drawn, c(1L, 3L))
})

test_that("nv_sample from a population array", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  pop <- nv_array(c(10, 20, 30))

  out <- nv_sample(x = pop, shape = 8L, initial_state = state, replace = TRUE)
  expect_equal(shape(out[[2]]), 8L)
  # dtype follows the population, not `dtype`
  expect_equal(dtype(out[[2]]), dtype(pop))
  expect_true(all(c(as_array(out[[2]])) %in% c(10, 20, 30)))

  # Without replacement, drawing everything permutes the population
  drawn <- c(as_array(nv_sample(x = pop, shape = 3L, initial_state = state)[[2]]))
  expect_setequal(drawn, c(10, 20, 30))

  # Weights select the heavy element nearly always
  weighted <- c(as_array(nv_sample(
    x = pop,
    shape = 200L,
    initial_state = state,
    replace = TRUE,
    probs = nv_array(c(0, 0, 1))
  )[[2]]))
  expect_true(all(weighted == 30))

  # `dtype` is meaningless when sampling from an array
  expect_error(
    nv_sample(x = pop, shape = 3L, initial_state = state, dtype = "i64"),
    "only applies when"
  )
  # Population must be 1-D
  expect_error(
    nv_sample(x = nv_array(matrix(1:6, nrow = 2)), shape = 3L, initial_state = state),
    "must be a single number or a 1-D array"
  )
  # Ambiguous plain R vectors are rejected
  expect_error(
    nv_sample(x = c(10, 20, 30), shape = 3L, initial_state = state),
    "Vectors of length > 1 are not allowed"
  )
  # `probs` must match the population size
  expect_error(
    nv_sample(x = pop, shape = 3L, initial_state = state, probs = nv_array(c(1, 2))),
    "must be a 1-D array of length 3"
  )
})

test_that("nv_sample composes inside jit", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  pop <- nv_array(c(10, 20, 30))

  # count form: `x` is a literal in the traced body
  f <- jit(function(s) nv_sample(8L, s, 6L, replace = TRUE))
  values <- c(as_array(f(state)[[2]]))
  expect_true(all(values >= 1L & values <= 6L))

  # population form: `x` and `probs` are traced inputs
  g <- jit(function(s, p, w) nv_sample(6L, s, p, replace = TRUE, probs = w))
  expect_true(all(c(as_array(g(state, pop, nv_array(c(0, 0, 1)))[[2]])) == 30))

  # without replacement, traced population
  h <- jit(function(s, p) nv_sample(3L, s, p))
  expect_setequal(c(as_array(h(state, pop)[[2]])), c(10, 20, 30))
})

test_that("nv_rng_state works the same across devices (eager)", {
  dev0 <- nv_device("cpu:0", "pjrt")
  dev1 <- nv_device("cpu:1", "pjrt")
  s0 <- nv_rng_state(42L, device = dev0)
  s1 <- nv_rng_state(42L, device = dev1)
  expect_equal(as_array(s0), as_array(s1))
})
