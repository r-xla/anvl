test_that("common_dtype_of: single argument", {
  expect_equal(common_dtype_of(AbstractArray("i32", Shape(c(1, 2)))), as_dtype("i32"))
  expect_equal(common_dtype_of(RDataArray(1.5, integer())), as_dtype("f32"))
})

test_that("common_dtype_of: two typed arguments", {
  check <- function(dt1, dt2, expected) {
    s1 <- AbstractArray(dt1, Shape(c(1, 2)))
    s2 <- AbstractArray(dt2, Shape(c(2, 1)))
    expect_equal(common_dtype_of(s1, s2), as_dtype(expected))
    expect_equal(common_dtype_of(s2, s1), as_dtype(expected))
  }

  check("i32", "i32", "i32")
  check("f32", "i32", "f32")
  check("f32", "f64", "f64")
  check("ui32", "i32", "i64")
})

test_that("common_dtype_of: an R value yields to a typed one", {
  check <- function(data, dt, expected) {
    rd <- RDataArray(data, integer())
    known <- AbstractArray(dt, Shape(c(2, 1)))
    expect_equal(common_dtype_of(rd, known), as_dtype(expected))
    expect_equal(common_dtype_of(known, rd), as_dtype(expected))
  }

  # It takes the dtype it meets, within its own category ...
  check(1L, "i8", "i8")
  check(1L, "i64", "i64")
  check(1.5, "f64", "f64")
  check(1.5, "f32", "f32")
  # ... crosses to the other category when that is what it meets ...
  check(1L, "f64", "f64")
  # ... but a float R value never becomes an integer, and nothing becomes a bool.
  check(1.5, "i32", "f32")
  check(1L, "bool", "i32")
  check(TRUE, "i32", "i32")
})

test_that("common_dtype_of: R values among themselves take their defaults", {
  check <- function(d1, d2, expected) {
    r1 <- RDataArray(d1, integer())
    r2 <- RDataArray(d2, integer())
    expect_equal(common_dtype_of(r1, r2), as_dtype(expected))
    expect_equal(common_dtype_of(r2, r1), as_dtype(expected))
  }
  check(1L, 2L, "i32")
  check(1.5, 2.5, "f32")
  check(1L, 2.5, "f32")
  check(TRUE, FALSE, "bool")
  check(TRUE, 1L, "i32")
  check(TRUE, 1.5, "f32")
})

test_that("common_dtype_of: multiple arguments", {
  i32 <- AbstractArray("i32", Shape(1))
  f32 <- AbstractArray("f32", Shape(2))
  f64 <- AbstractArray("f64", Shape(3))

  expect_equal(common_dtype_of(i32, f32, f64), as_dtype("f64"))
  expect_equal(common_dtype_of(f64, f32, i32), as_dtype("f64"))
  expect_equal(common_dtype_of(i32, i32, i32), as_dtype("i32"))
  # An R value in the middle still yields to the typed ones around it.
  expect_equal(common_dtype_of(RDataArray(1L, integer()), AbstractArray("i64", Shape(2))), as_dtype("i64"))
  expect_equal(
    common_dtype_of(RDataArray(1.5, integer()), i32, AbstractArray("f64", Shape(1))),
    as_dtype("f64")
  )
})

test_that("common_dtype_of: error on no arguments", {
  expect_error(common_dtype_of(), "No arguments provided")
})

test_that("promote_dt_known", {
  check <- function(dt1, dt2, dt3) {
    expect_equal(
      promote_dt_known(as_dtype(dt1), as_dtype(dt2)),
      as_dtype(dt3)
    )
    expect_equal(
      promote_dt_known(as_dtype(dt2), as_dtype(dt1)),
      as_dtype(dt3)
    )
  }

  check("f64", "f64", "f64")
  check("i32", "i32", "i32")
  check("bool", "bool", "bool")

  # floats dominate
  check("f64", "f32", "f64")
  check("f64", "i32", "f64")
  check("f32", "i32", "f32")
  check("f32", "bool", "f32")

  # signed ints
  check("i32", "i16", "i32")
  check("i64", "i32", "i64")
  check("i64", "i16", "i64")
  check("i64", "bool", "i64")
  # against unsigned ints
  check("i32", "ui8", "i32")
  check("i32", "ui32", "i64")
  check("i64", "ui64", "i64")
  # unsigned vs unsigned
  check("ui64", "ui32", "ui64")
})

test_that("promote_dt_rdata", {
  check <- function(rdtype, known, z) {
    expect_equal(
      promote_dt_rdata(as_dtype(rdtype), as_dtype(known)),
      as_dtype(z)
    )
  }
  # An R value commits to i32, f32 or bool, and yields from there.
  check("i32", "i32", "i32")
  check("bool", "bool", "bool")
  check("f32", "f32", "f32")

  check("i32", "i8", "i8")
  check("i32", "i64", "i64")
  check("i32", "bool", "i32")

  check("f32", "f64", "f64")
  check("f32", "i32", "f32")

  check("bool", "f32", "f32")
  check("bool", "i32", "i32")
})

test_that("common_dtype is the promotion of two known dtypes", {
  expect_equal(common_dtype("i32", "f32"), as_dtype("f32"))
  expect_equal(common_dtype("i32", "i64"), as_dtype("i64"))
  expect_equal(common_dtype("f64", "f32"), as_dtype("f64"))
})
