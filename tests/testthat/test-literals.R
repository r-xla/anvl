# REVIEW: test-literals should be integrated into test-rdata as literals are rdata.
test_that("literals are downcast if possible", {
  # literal (has default type i32 for ints) is downcast to i16
  f1 <- function(x) {
    x * 2L
  }
  expect_equal(
    jit(f1)(nv_scalar(1, dtype = "i16")),
    nv_scalar(2L, dtype = "i16")
  )
})

test_that("can combine literals", {
  f1 <- function() {
    nv_mul(2, 3)
  }
  expect_equal(
    jit(f1)(),
    nv_scalar(6, dtype = "f32")
  )
  f2 <- function() {
    nv_mul(2, 3L)
  }
  expect_equal(
    jit(f2)(),
    nv_scalar(6, dtype = "f32")
  )
})

test_that("literals can be upcast", {
  f1 <- function(x) {
    nv_mul(2, x)
  }
  expect_equal(
    jit(f1)(nv_scalar(3, dtype = "f64")),
    nv_scalar(6, dtype = "f64")
  )
})

test_that("literals can be returned in jit", {
  f2 <- function() {
    1
  }
  expect_equal(
    jit(f2)(),
    nv_scalar(1, dtype = "f32")
  )
})

test_that("literals work in gradient", {
  f <- function(x) {
    x * 2
  }
  grad <- jit(gradient(f, wrt = "x"))
  expect_equal(
    grad(nv_scalar(1)),
    list(x = nv_scalar(2))
  )
})

test_that("literals work in infix functions", {
  f1 <- function(x) {
    x * 2
  }
  expect_equal(
    jit(f1)(nv_scalar(1)),
    nv_scalar(2)
  )
  f2 <- function(x) {
    2 + x
  }
  expect_equal(
    jit(f2)(nv_scalar(1)),
    nv_scalar(3)
  )
})

test_that("can use integers, logicals and doubles", {
  f1 <- function(x) {
    x * 2L
  }
  expect_equal(
    jit(f1)(nv_scalar(1L)),
    nv_scalar(2L)
  )
  f2 <- function(x) {
    x * 2
  }
  expect_equal(
    jit(f2)(nv_scalar(1.0)),
    nv_scalar(2.0)
  )
  f3 <- function(x) {
    # jarl-ignore redundant_equals: testing literal TRUE comparison
    x == TRUE
  }
  expect_equal(
    jit(f3)(nv_scalar(TRUE)),
    nv_scalar(TRUE)
  )
  expect_equal(
    jit(f3)(nv_scalar(FALSE)),
    nv_scalar(FALSE)
  )
})
