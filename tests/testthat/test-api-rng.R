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

test_that("nv_rdunif", {
  # statistical validity checks are in inst/random

  # Test with equal probabilities
  out1 <- nv_rdunif(n = 6L, shape = 10L, initial_state = nv_array(c(1, 2), dtype = "ui64"))

  expect_equal(shape(out1[[1]]), 2L)
  expect_equal(shape(out1[[2]]), 10L)
  expect_equal(dtype(out1[[2]]), as_dtype("i32"))

  # All values should be in 1:6
  values1 <- c(as_array(out1[[2]]))
  expect_true(all(values1 >= 1L & values1 <= 6L))

  # Test 2D output shape
  out3 <- nv_rdunif(n = 4L, shape = c(2L, 3L), initial_state = nv_array(c(1, 2), dtype = "ui64"))
  expect_equal(shape(out3[[2]]), c(2L, 3L))
})

test_that("nv_rng_state works the same across devices (eager)", {
  dev0 <- nv_device("cpu:0", "xla")
  dev1 <- nv_device("cpu:1", "xla")
  s0 <- nv_rng_state(42L, device = dev0)
  s1 <- nv_rng_state(42L, device = dev1)
  expect_equal(as_array(s0), as_array(s1))
})
