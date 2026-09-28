test_that("expect_dtype accepts a string and a DataType", {
  expect_success(expect_dtype(nv_scalar(1L, dtype = "i32"), "i32"))
  expect_success(expect_dtype(nv_scalar(1L, dtype = "i32"), as_dtype("i32")))
  expect_success(expect_dtype(nv_scalar(1L), default_int()))
  expect_success(expect_dtype(nv_scalar(1L, dtype = "i8"), dtype(nv_scalar(2L, dtype = "i8"))))
})

test_that("expect_dtype names the expression and both data types on failure", {
  out <- nv_scalar(1L, dtype = "i16")
  expect_failure(expect_dtype(out, "i32"), 'dtype(`out`) is "i16", not "i32".', fixed = TRUE)
})

test_that("expect_dtype passes an R value on to dtype(), which rejects it", {
  # An R value has no data type of its own; `peek_dtype()` is the question to
  # ask about one, so the expectation must not answer it silently.
  expect_error(expect_dtype(1.5, "f32"), "no data type of its own")
})

test_that("expect_dtype returns its input invisibly", {
  x <- nv_scalar(1L)
  expect_identical(withVisible(expect_dtype(x, default_int())), list(value = x, visible = FALSE))
})

test_that("expect_shape compares axis sizes as integers", {
  x <- nv_array(array(1:6, c(2, 3)))
  expect_success(expect_shape(x, c(2, 3)))
  expect_success(expect_shape(x, c(2L, 3L)))
  expect_success(expect_shape(nv_scalar(1L), integer()))
  expect_success(expect_shape(x, shape(x)))
})

test_that("expect_shape names the expression and both shapes on failure", {
  out <- nv_array(array(1:6, c(2, 3)))
  expect_failure(expect_shape(out, c(3, 2)), "shape(`out`) is (2x3), not (3x2).", fixed = TRUE)
  expect_failure(expect_shape(out, integer()), "shape(`out`) is (2x3), not ().", fixed = TRUE)
})
