test_that("jit: autoconverts length-1 numeric scalar", {
  f <- jit(identity)
  out <- f(1)
  expect_equal(out, nv_scalar(1))
})

test_that("jit: autoconverted scalar + literal takes the default dtype", {
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
  f <- jit(identity)
  out <- f(TRUE)
  # A logical is the one R type that names its dtype: there is nothing for
  # `TRUE` to become other than `bool`.
  expect_equal(out, nv_scalar(TRUE))
})

test_that("jit: promotes scalar dtype against a typed scalar", {
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

test_that("jit: an autoconverted leaf is not baked into the program", {
  f <- jit(function(x, y) x + y)
  x <- nv_scalar(0, dtype = "f64")
  expect_identical(as_array(f(x, sqrt(2))), sqrt(2))
  # Same key, so this is a cache hit -- and it must still return its own value,
  # not the one the program was compiled with.
  expect_identical(as_array(f(x, pi)), pi)
  expect_equal(cache_size(f), 1L)
})

test_that("jit: a leaf used at two dtypes is exact at the wider one", {
  f <- jit(function(t) {
    list(
      wide = nv_scalar(0, dtype = "f64") + t,
      narrow = nv_scalar(0, dtype = "f32") + t
    )
  })
  out <- f(sqrt(2))
  expect_equal(dtype(out$wide), as_dtype("f64"))
  expect_equal(dtype(out$narrow), as_dtype("f32"))
  # Uploaded once as f64 and converted down for the f32 site: one rounding,
  # exactly as an f32 upload would have been.
  expect_identical(as_array(out$wide), sqrt(2))
  expect_identical(as_array(out$narrow), as_array(nv_scalar(sqrt(2), dtype = "f32")))
})

test_that("jit: an unused leaf is still an input the call supplies", {
  f <- jit(function(x, y) x + 1)
  expect_identical(as_array(f(nv_scalar(1, dtype = "f64"), 99)), 2)
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

test_that("quickr: an autoconverted leaf reaches an f64 use site exactly", {
  skip_if_no_quickr()
  local_backend("quickr")
  x <- nv_scalar(1, dtype = "f64")
  expect_identical(as_array(jit(function(x) x / sqrt(2))(x)), 1 / sqrt(2))
  expect_identical(as_array(jit(function(x, y) x / y)(x, sqrt(2))), 1 / sqrt(2))
})

test_that("quickr: an autoconverted leaf is coerced to the program's dtype", {
  skip_if_no_quickr()
  local_backend("quickr")
  # The leaf arrives as an R integer but the program consumes it as f64.
  f <- jit(function(x, y) x + y)
  expect_identical(as_array(f(nv_scalar(1, dtype = "f64"), 2L)), 3)
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
