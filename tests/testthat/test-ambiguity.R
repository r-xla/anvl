test_that("nv_scalar returns non-ambiguous array by default", {
  # Numeric scalars without explicit dtype should be non-ambiguous
  x_f32 <- nv_scalar(1.0)
  expect_false(ambiguous(x_f32))
  expect_equal(dtype(x_f32), as_dtype("f32"))

  # Integer scalars without explicit dtype should be non-ambiguous
  x_i32 <- nv_scalar(1L)
  expect_false(ambiguous(x_i32))
  expect_equal(dtype(x_i32), as_dtype("i32"))

  # Logical scalars are also non-ambiguous
  x_pred <- nv_scalar(TRUE)
  expect_false(ambiguous(x_pred))
  expect_equal(dtype(x_pred), as_dtype("bool"))
})

test_that("nv_scalar returns non-ambiguous array when dtype is specified", {
  x_f32 <- nv_scalar(1.0, dtype = "f32")
  expect_false(ambiguous(x_f32))
  expect_equal(dtype(x_f32), as_dtype("f32"))

  x_i32 <- nv_scalar(1L, dtype = "i32")
  expect_false(ambiguous(x_i32))
  expect_equal(dtype(x_i32), as_dtype("i32"))

  x_f64 <- nv_scalar(1.0, dtype = "f64")
  expect_false(ambiguous(x_f64))
  expect_equal(dtype(x_f64), as_dtype("f64"))
})

test_that("nv_array returns non-ambiguous array", {
  # nv_array always creates non-ambiguous arrays
  x <- nv_array(1:4)
  expect_false(ambiguous(x))

  y <- nv_array(c(1.0, 2.0, 3.0))
  expect_false(ambiguous(y))
})

test_that("JIT propagates ambiguity from inputs to outputs", {
  f <- jit(function(x, y) x + y)

  # Both inputs ambiguous -> output should be ambiguous
  x_amb <- nv_scalar(1.0, ambiguous = TRUE)
  y_amb <- nv_scalar(2.0, ambiguous = TRUE)
  result_amb <- f(x_amb, y_amb)
  expect_true(ambiguous(result_amb))
  expect_equal(result_amb, nv_scalar(3.0, ambiguous = TRUE))

  # One input non-ambiguous -> output should be non-ambiguous
  x_nonamb <- nv_scalar(1.0, dtype = "f32")
  y_amb2 <- nv_scalar(2.0, ambiguous = TRUE)
  result_mixed <- f(x_nonamb, y_amb2)
  expect_false(ambiguous(result_mixed))
  expect_equal(result_mixed, nv_scalar(3.0, dtype = "f32"))

  # Both inputs non-ambiguous -> output should be non-ambiguous
  x_nonamb2 <- nv_scalar(1.0, dtype = "f32")
  y_nonamb <- nv_scalar(2.0, dtype = "f32")
  result_nonamb <- f(x_nonamb2, y_nonamb)
  expect_false(ambiguous(result_nonamb))
  expect_equal(result_nonamb, nv_scalar(3.0, dtype = "f32"))
})

test_that("JIT preserves ambiguity through unary operations", {
  f <- jit(function(x) -x)

  # Ambiguous input -> ambiguous output
  x_amb <- nv_scalar(1.0, ambiguous = TRUE)
  result_amb <- f(x_amb)
  expect_true(ambiguous(result_amb))
  expect_equal(result_amb, nv_scalar(-1.0, ambiguous = TRUE))

  # Non-ambiguous input -> non-ambiguous output
  x_nonamb <- nv_scalar(1.0, dtype = "f32")
  result_nonamb <- f(x_nonamb)
  expect_false(ambiguous(result_nonamb))
  expect_equal(result_nonamb, nv_scalar(-1.0, dtype = "f32"))
})

test_that("JIT handles constants with ambiguity", {
  # Constant created with non-ambiguous nv_scalar (new default)
  f <- jit(function(x) x + nv_scalar(10))

  x_amb <- nv_scalar(1.0, ambiguous = TRUE)
  result <- f(x_amb)
  # Input is ambiguous, constant is non-ambiguous -> non-ambiguous
  expect_false(ambiguous(result))
  expect_equal(result, nv_scalar(11.0))

  # Non-ambiguous input with non-ambiguous constant
  x_nonamb <- nv_scalar(1.0, dtype = "f32")
  result2 <- f(x_nonamb)
  # Both non-ambiguous -> non-ambiguous
  expect_false(ambiguous(result2))
  expect_equal(result2, nv_scalar(11.0, dtype = "f32"))
})

test_that("format.AnvlArray shows ambiguity with ?", {
  x_amb <- nv_scalar(1.0, ambiguous = TRUE)
  expect_match(format(x_amb), "f32\\?")

  x_nonamb <- nv_scalar(1.0, dtype = "f32")
  expect_match(format(x_nonamb), "f32[^?]|f32$")
})

test_that("boolean reductions never produce ambiguous output", {
  f_any <- jit(function(x) prim_reduce_any(x, axes = 1L))
  f_all <- jit(function(x) prim_reduce_all(x, axes = 1L))

  # Even with ambiguous input, output is never ambiguous
  x_amb <- nv_array(c(TRUE, FALSE), ambiguous = TRUE)
  expect_true(ambiguous(x_amb))
  expect_false(ambiguous(f_any(x_amb)))
  expect_false(ambiguous(f_all(x_amb)))
})

test_that("ambiguous() generic works for AnvlArray and AbstractArray", {
  # AnvlArray
  x_anvl_amb <- nv_scalar(1.0, ambiguous = TRUE)
  expect_true(ambiguous(x_anvl_amb))

  x_anvl_nonamb <- nv_array(1.0)
  expect_false(ambiguous(x_anvl_nonamb))

  # AbstractArray
  aval_amb <- nv_aval("f32", c(2, 3), ambiguous = TRUE)
  expect_true(ambiguous(aval_amb))

  aval_nonamb <- nv_aval("f32", c(2, 3), ambiguous = FALSE)
  expect_false(ambiguous(aval_nonamb))
})
