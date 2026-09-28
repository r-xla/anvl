test_that("nv_rnorm", {
  # statistical validity checks are in inst/random
  out <- nv_rnorm(nv_array(c(1, 2), dtype = "ui64"), dtype = "f32", shape = c(2, 3))
  expect_dtype(out[[1]], "ui64")
  expect_shape(out[[1]], 2L)
  expect_dtype(out[[2]], "f32")
  expect_shape(out[[2]], c(2L, 3L))

  # test with uneven total number of RVs
  out <- nv_rnorm(nv_array(c(1, 2), dtype = "ui64"), dtype = "f32", shape = c(3, 3))
  expect_shape(out[[2]], c(3L, 3L))

  # test mean/sd parameters with small sample
  out <- nv_rnorm(
    nv_array(c(3, 83), dtype = "ui64"),
    dtype = "f64",
    shape = c(2L, 3L),
    mean = 10,
    sd = 9
  )
  expect_dtype(out[[1]], "ui64")
  expect_shape(out[[1]], 2L)
  expect_shape(out[[2]], c(2L, 3L))
  expect_dtype(out[[2]], "f64")
})

test_that("nv_rnorm accepts arrayish mean and sd", {
  state <- nv_array(c(1, 2), dtype = "ui64")

  # An elementwise mean of the same shape as the sample
  means <- nv_array(matrix(c(-1000, 1000, -1000, 1000, -1000, 1000), nrow = 2))
  out <- nv_rnorm(c(2, 3), state, dtype = "f64", mean = means, sd = 1)
  values <- as_array(out[[2]])
  expect_shape(out[[2]], c(2L, 3L))
  # sd is 1, so each draw stays near its own mean
  expect_true(all(values[1, ] < -900))
  expect_true(all(values[2, ] > 900))

  # An elementwise sd
  sds <- nv_array(matrix(c(1e-6, 1e6, 1e-6, 1e6, 1e-6, 1e6), nrow = 2))
  spread <- as_array(nv_rnorm(c(2, 3), state, dtype = "f64", sd = sds)[[2]])
  expect_true(all(abs(spread[1, ]) < 1))
  expect_true(all(abs(spread[2, ]) > 1))

  # mean/sd may be traced under jit
  f <- jit(function(s, m, sdev) nv_rnorm(c(2, 3), s, dtype = "f64", mean = m, sd = sdev))
  traced <- as_array(f(state, nv_scalar(1000, dtype = "f64"), nv_scalar(1, dtype = "f64"))[[2]])
  expect_true(all(traced > 900))

  # An odd number of draws still reshapes correctly with arrayish mean
  odd_means <- nv_array(matrix(rep(c(-1000, 0, 1000), each = 3), nrow = 3))
  odd <- as_array(nv_rnorm(c(3, 3), state, dtype = "f64", mean = odd_means)[[2]])
  expect_true(all(odd[, 1] < -900) && all(abs(odd[, 2]) < 100) && all(odd[, 3] > 900))

  # Anything else than a scalar or the sample's shape is refused, including for
  # a scalar sample, which would otherwise take the shape of `mean`/`sd`
  expect_error(
    nv_rnorm(c(2, 3), state, mean = nv_array(matrix(0, 2, 1))),
    "must be a scalar or have the shape of the sample"
  )
  expect_error(
    nv_rnorm(integer(), state, sd = nv_array(c(1, 2, 3))),
    "must be a scalar or have the shape of the sample"
  )
})

test_that("rng rejects non-f32/f64 dtypes", {
  key <- nv_array(c(1, 2), dtype = "ui64")
  # A float, but not one the bit manipulation can build.
  expect_error(
    nv_rnorm(key, dtype = "bf16", shape = 2L),
    "must be a 32- or 64-bit float data type"
  )
  expect_error(
    nv_rnorm(key, dtype = "i32", shape = 2L),
    "must be a float data type"
  )
})

test_that("nv_runif", {
  # statistical validity checks are in inst/random
  out <- nv_runif(
    nv_array(c(1, 2), dtype = "ui64"),
    dtype = "f32",
    shape = c(3, 4),
    min = -1,
    max = 1
  )

  expect_dtype(out[[1]], "ui64")
  expect_shape(out[[1]], 2L)
  expect_shape(out[[2]], c(3L, 4L))
  expect_dtype(out[[2]], "f32")
})

test_that("nv_rbinom", {
  # statistical validity checks are in inst/random
  out <- nv_rbinom(nv_array(c(1, 2), dtype = "ui64"), dtype = "i32", shape = c(2, 5))

  expect_shape(out[[1]], 2L)
  expect_shape(out[[2]], c(2L, 5L))
  expect_dtype(out[[2]], "i32")

  # All values should be 0 or 1
  values <- as.vector(out[[2]])
  expect_true(all(values %in% c(0L, 1L)))

  # Test with different dtype
  out2 <- nv_rbinom(nv_array(c(1, 2), dtype = "ui64"), dtype = "f32", shape = 10L)
  expect_dtype(out2[[2]], "f32")
  expect_shape(out2[[2]], 10L)

  # Test with non-multiple-of-8 shape (tests slicing)
  out3 <- nv_rbinom(nv_array(c(1, 2), dtype = "ui64"), dtype = "i32", shape = c(3, 3))
  expect_shape(out3[[2]], c(3L, 3L))
})

test_that("nv_runif with min == max returns the pair, state advanced", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  out <- nv_runif(c(2, 3), state, min = 5, max = 5)
  expect_named(out, c("state", "values"))
  # `min`/`max` may be traced, so the draw is made and the state advanced as
  # for any other interval.
  expect_equal(as.vector(out$state), as.vector(nv_runif(c(2, 3), state)$state))
  expect_shape(out$values, c(2L, 3L))
  expect_true(all(as.vector(out$values) == 5))
})

test_that("nv_runif accepts arrayish min and max", {
  state <- nv_array(c(1, 2), dtype = "ui64")

  # An elementwise interval of the same shape as the sample
  lower <- nv_array(matrix(c(0, 10, 100, 1000, 10000, 100000), nrow = 2), dtype = "f64")
  out <- nv_runif(c(2, 3), state, min = lower, max = lower + 1)
  values <- as_array(out$values)
  expect_shape(out$values, c(2L, 3L))
  expect_dtype(out$values, "f64")
  expect_true(all(values > as_array(lower) & values < as_array(lower) + 1))

  # A scalar bound combines with an array one, and the draws are those of the
  # standard uniform, scaled and shifted elementwise
  upper <- nv_array(matrix(1:6, nrow = 2), dtype = "f64")
  u <- as_array(nv_runif(c(2, 3), state, dtype = "f64")$values)
  expect_equal(as_array(nv_runif(c(2, 3), state, min = 0, max = upper)$values), u * 1:6)

  # min/max may be traced under jit
  f <- jit(function(s, a, b) nv_runif(c(2, 3), s, min = a, max = b))
  traced <- as_array(f(state, nv_scalar(1000, dtype = "f64"), nv_scalar(1001, dtype = "f64"))$values)
  expect_true(all(traced > 1000 & traced < 1001))

  # Anything else than a scalar or the sample's shape is refused
  expect_error(
    nv_runif(c(2, 3), state, max = nv_array(matrix(1, 2, 1))),
    "must be a scalar or have the shape of the sample"
  )
  expect_error(
    nv_runif(integer(), state, min = nv_array(c(0, 1))),
    "must be a scalar or have the shape of the sample"
  )
})

test_that("nv_runif gives NaN for an invalid interval, like runif()", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  lower <- c(0, 5, 1, -Inf, NaN, 10)
  upper <- c(1, 5, 0, 1, 1, Inf)
  out <- nv_runif(6L, state, dtype = "f64", min = nv_array(lower), max = nv_array(upper))
  expect_equal(is.nan(as.vector(out$values)), suppressWarnings(is.nan(runif(6L, lower, upper))))
  expect_equal(as.vector(out$values)[2L], 5)
})

test_that("nv_runif differentiates with respect to min and max", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  u <- as.vector(nv_runif(c(2, 3), state, dtype = "f64")$values)
  g <- jit(gradient(function(a, b) sum(nv_runif(c(2, 3), state, min = a, max = b)$values)))
  grads <- g(nv_array(matrix(-2, 2, 3), dtype = "f64"), nv_array(matrix(3, 2, 3), dtype = "f64"))
  expect_equal(as.vector(grads[[1L]]), 1 - u)
  expect_equal(as.vector(grads[[2L]]), u)
})

test_that("nv_rbinom and nv_sample_int reject a boolean data type", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  expect_error(
    nv_rbinom(state, dtype = "bool", shape = 4L),
    "must be a numeric data type"
  )
})

test_that("nv_sample_int", {
  # statistical validity checks are in inst/random
  state <- nv_array(c(1, 2), dtype = "ui64")

  out1 <- nv_sample_int(n = 6L, shape = 10L, state = state)

  expect_shape(out1[[1]], 2L)
  expect_shape(out1[[2]], 10L)
  expect_dtype(out1[[2]], default_int())

  # All values should be in 1:6
  values1 <- as.vector(out1[[2]])
  expect_true(all(values1 >= 1L & values1 <= 6L))

  # Test 2D output shape
  out3 <- nv_sample_int(n = 4L, shape = c(2L, 3L), state = state)
  expect_shape(out3[[2]], c(2L, 3L))

  # The last integer is reachable and the first is not over-represented
  values4 <- as.vector(nv_sample_int(n = 6L, shape = 5000L, state = state)[[2]])
  expect_setequal(unique(values4), 1:6)
  expect_true(all(abs(as.numeric(table(values4)) / 5000 - 1 / 6) < 0.02))

  # The dtype of the drawn integers is configurable
  out5 <- nv_sample_int(n = 6L, shape = 4L, state = state, dtype = "i64")
  expect_dtype(out5[[2]], "i64")

  # A population of size one is always drawn
  expect_true(all(as.vector(nv_sample_int(n = 1L, shape = 20L, state = state)[[2]]) == 1L))
})

test_that("nv_sample from a population array", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  pop <- nv_array(c(10, 20, 30))

  out <- nv_sample(x = pop, shape = 8L, state = state)
  expect_shape(out[[2]], 8L)
  # The result has the data type of the population
  expect_dtype(out[[2]], dtype(pop))
  expect_true(all(as.vector(out[[2]]) %in% c(10, 20, 30)))

  # 2D output shape
  expect_shape(nv_sample(x = pop, shape = c(2L, 3L), state = state)[[2]], c(2L, 3L))

  # Every element of the population is reachable
  many <- as.vector(nv_sample(x = pop, shape = 500L, state = state)[[2]])
  expect_setequal(unique(many), c(10, 20, 30))

  # Unlike R's `sample()`, a length-one population is not a count
  expect_true(all(as.vector(nv_sample(x = nv_array(6), shape = 5L, state = state)[[2]]) == 6))

  # Population must be 1-D
  expect_error(
    nv_sample(x = nv_array(matrix(1:6, nrow = 2)), shape = 3L, state = state),
    "must be a 1-D array"
  )
})

test_that("nv_sample and nv_sample_int compose inside jit", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  pop <- nv_array(c(10, 20, 30))

  # `n` is static, so it may be a literal in the traced body
  f <- jit(function(s) nv_sample_int(8L, s, 6L))
  values <- as.vector(f(state)[[2]])
  expect_true(all(values >= 1L & values <= 6L))

  # the population is a traced input
  g <- jit(function(s, p) nv_sample(6L, s, p))
  expect_true(all(as.vector(g(state, pop)[[2]]) %in% c(10, 20, 30)))
})

test_that("nv_rng_state works the same across devices (eager)", {
  dev0 <- nv_device("cpu:0")
  dev1 <- nv_device("cpu:1")
  s0 <- nv_rng_state(42L, device = dev0)
  s1 <- nv_rng_state(42L, device = dev1)
  expect_equal(as_array(s0), as_array(s1))
})

test_that("nv_rnorm takes the sample's dtype from mean and sd", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  draw <- function(...) dtype(nv_rnorm(2L, state, ...)[[2L]])

  # Neither brings a data type, so the sample falls back to the default float
  # rather than to whatever R stores its numbers as.
  expect_equal(draw(), default_float())
  expect_equal(draw(mean = 0L, sd = 1L), default_float())

  # Either one that has a data type gives the sample its own.
  expect_equal(draw(mean = nv_scalar(1, dtype = "f64")), as_dtype("f64"))
  expect_equal(draw(sd = nv_scalar(1, dtype = "f64")), as_dtype("f64"))
  expect_equal(
    draw(mean = nv_scalar(1, dtype = "f64"), sd = nv_scalar(1, dtype = "f32")),
    as_dtype("f64")
  )
  # An R value keeps the sample a float even where the other is an integer.
  expect_equal(draw(mean = nv_scalar(1L)), default_float())

  # `dtype` is the caller's word over the arguments', and is refused where the
  # sample could not hold them.
  expect_equal(draw(dtype = "f64"), as_dtype("f64"))
  expect_error(
    draw(dtype = "f32", mean = nv_scalar(1, dtype = "f64")),
    "Cannot bring `mean` to data type \"f32\""
  )

  # Arguments that agree on a data type the generator cannot draw at say so.
  expect_error(
    draw(mean = nv_scalar(1L), sd = nv_scalar(2L)),
    "must be a float data type"
  )
  expect_error(draw(mean = nv_scalar(1L), sd = nv_scalar(2L)), "Pass `dtype`")
})
