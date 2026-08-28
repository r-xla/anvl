# REVIEW: This should probably also be integrated into test-rdata.R
# Probably can deduplicte quite a bit
test_that("jit: autoconverts length-1 numeric scalar", {
  f <- jit(identity)
  out <- f(1)
  expect_equal(out, nv_scalar(1))
})

test_that("jit: an autoconverted scalar plus a literal takes the default dtype", {
  f <- jit(\(x) x + 1)
  out <- f(1)
  expect_equal(dtype(out), as_dtype("f32"))
  expect_equal(shape(out), integer())
})

test_that("jit: autoconverts length-1 integer scalar", {
  f <- jit(identity)
  out <- f(1L)
  expect_equal(out, nv_scalar(1L))
})

test_that("jit: autoconverts length-1 logical scalar", {
  # A logical is the one R type that names its dtype: there is nothing for
  # `TRUE` to become other than `bool`.
  f <- jit(identity)
  out <- f(TRUE)
  expect_equal(out, nv_scalar(TRUE))
})

test_that("jit: an R scalar takes the dtype of the typed array it meets", {
  f <- jit(\(x, y) x + y)
  out <- f(1, nv_scalar(2, dtype = "f64"))
  expect_equal(dtype(out), as_dtype("f64"))
})

test_that("jit: autoconverts matrix via nv_array", {
  f <- jit(identity)
  out <- f(matrix(1:4, 2, 2))
  expect_equal(out, nv_matrix(1:4, nrow = 2, ncol = 2))
})

test_that("jit: autoconverts higher-axis array via nv_array", {
  f <- jit(identity)
  a <- array(1:24, dim = c(2, 3, 4))
  out <- f(a)
  expect_equal(dtype(out), as_dtype("i32"))
  expect_equal(shape(out), c(2L, 3L, 4L))
})

test_that("jit: bare vector without dim errors", {
  f <- jit(function(x) x)
  expect_snapshot(f(c(1, 2, 3)), error = TRUE)
})

test_that("jit: non-array/non-scalar leaves (e.g. character) error", {
  f <- jit(function(x) x)
  expect_snapshot(f("hello"), error = TRUE)
})

test_that("jit: nested list is flattened; leaves are autoconverted", {
  f <- jit(function(pair) pair[[1]] + pair[[2]])
  out <- f(list(1, 2))
  expect_equal(dtype(out), as_dtype("f32"))
  expect_equal(shape(out), integer())
  expect_equal(as_array(out), 3)
})

test_that("jit: static args are not autoconverted", {
  f <- jit(function(x, flag) if (flag) x + 1 else x * 2, static = "flag")
  out <- f(nv_scalar(3), TRUE)
  expect_equal(as_array(out), 4)
  out2 <- f(3, FALSE)
  expect_equal(as_array(out2), 6)
})

test_that("jit: inside trace, autoconvert does not fire on GraphValues", {
  inner <- jit(function(x) x + 1)
  outer <- jit(function(x) inner(x))
  out <- outer(nv_scalar(1))
  expect_equal(as_array(out), 2)
})

test_that("jit_eval: scalar expression works unchanged", {
  expect_equal(as_array(jit_eval(nv_scalar(1) + nv_scalar(2))), 3)
})

test_that("quickr: autoconverts scalar input", {
  skip_if_no_quickr()
  local_backend("quickr")
  f <- jit(identity)
  out <- f(1)
  expect_equal(out, nv_scalar(1))
})

test_that("quickr: autoconverts matrix input", {
  skip_if_no_quickr()
  local_backend("quickr")
  f <- jit(identity)
  out <- f(matrix(1:4, 2, 2))
  expect_equal(dtype(out), as_dtype("i32"))
  expect_equal(shape(out), c(2L, 2L))
  expect_equal(out, nv_matrix(1:4, nrow = 2, ncol = 2))
})

test_that("quickr: nested input tree with mixed AnvlArray/scalar still works", {
  skip_if_no_quickr()
  local_backend("quickr")
  f <- jit(function(pair) pair[[1]] + pair[[2]])
  out <- f(list(nv_scalar(1L), 2L))
  expect_equal(out, nv_scalar(3L))
})

test_that("quickr: bare vector errors", {
  skip_if_no_quickr()
  local_backend("quickr")
  f <- jit(function(x) x)
  expect_snapshot(f(c(1, 2, 3)), error = TRUE)
})

test_that("jit: error shows path for nested list element", {
  f <- jit(function(l) l[[1]])
  expect_snapshot(f(list(list(a = "abc"))), error = TRUE)
})

test_that("jit: error shows path for unnamed nested element", {
  f <- jit(function(pair) pair[[1]])
  expect_snapshot(f(list("bad", nv_scalar(1))), error = TRUE)
})
