test_that("an R value takes the dtype of the array it meets", {
  f <- function(x) x * 1L
  expect_equal(
    jit(f)(nv_scalar(2L, "i16")),
    nv_scalar(2L, dtype = "i16")
  )
  expect_equal(
    jit(f)(nv_scalar(2, "f64")),
    nv_scalar(2, dtype = "f64")
  )
})

test_that("an R value that meets nothing commits to the default dtype", {
  expect_equal(jit(function() 1 * 2)(), nv_scalar(2, dtype = "f32"))
  expect_equal(jit(function() 1L * 2L)(), nv_scalar(2L, dtype = "i32"))
})

test_that("a value that has committed keeps its dtype", {
  # `x * 1L` commits to i32 -- the R value cannot stay a bool -- and the i16
  # then promotes against a real i32, which wins.
  f <- function(x, y) (x * 1L) + y
  expect_equal(
    jit(f)(nv_scalar(TRUE), nv_scalar(2L, "i16")),
    nv_scalar(3L, dtype = "i32")
  )
})

test_that("prim_convert reverse", {
  out <- jit(function(x) {
    z <- prim_convert(x, "f32")
    a <- gradient(\(y) {
      y_int <- prim_convert(y, "i32")
      prim_convert(y_int, "f32")
    })(z)[[1L]]
  })(nv_scalar(TRUE))
  expect_equal(out, nv_scalar(1, dtype = "f32"))
})

test_that("prim_if outputs keep the dtype of their branches", {
  f <- function(pred, x) {
    x <- x * 2L
    nv_if(pred, \() x, \() x * x) * nv_scalar(3L, dtype = "i16")
  }
  expect_equal(
    jit(f)(nv_scalar(TRUE), nv_scalar(TRUE)),
    nv_scalar(6L, dtype = "i32")
  )
})

test_that("prim_while carries the dtype of its state", {
  f <- jit(function(n) {
    i <- 0L * nv_scalar(TRUE)
    nv_while(list(i = i), \(i) i <= n, \(i) {
      i <- i + 1L
      list(i = i)
    })[[1L]] *
      nv_scalar(3L, dtype = "i16")
  })
  expect_equal(
    f(nv_scalar(10L)),
    nv_scalar(33L, dtype = "i32")
  )
})

test_that("a logical R value is a bool, not an uncommitted value", {
  f <- function(x) x * TRUE
  graph <- trace_fn(f, list(x = nv_scalar(1L)))
  expect_equal(dtype(graph$calls[[1L]]$inputs[[2L]]$aval), as_dtype("i32"))
})
