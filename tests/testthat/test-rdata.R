# An R value entering a program has no dtype of its own: it is built into the
# program at the dtype its use site needs, from the R data itself. These tests
# assert concrete numbers rather than comparing against another framework, so a
# missing optional dependency cannot silently skip a precision check.

test_that("an R literal in the body reaches an f64 use site exactly", {
  x <- nv_scalar(1, dtype = "f64")
  expect_identical(as_array(jit(function(x) x / sqrt(2))(x)), 1 / sqrt(2))
  expect_identical(as_array(jit(function(x) x * pi)(x)), pi)
  expect_identical(as_array(jit(function(x) x + 0.1)(x)), 1.1)
  expect_identical(as_array(jit(function(x) x - 0.1)(x)), 0.9)
})

test_that("an R literal reaches an f64 use site exactly in eager mode too", {
  # Every nv_* function is jit-wrapped, so here the literal is an *argument* of
  # the call and is uploaded at the dtype the program decided on.
  x <- nv_scalar(1, dtype = "f64")
  expect_identical(as_array(x / sqrt(2)), 1 / sqrt(2))
  expect_identical(as_array(x * pi), pi)
  expect_identical(as_array(x + 0.1), 1.1)
  expect_identical(as_array(x - 0.1), 0.9)
  # ... and with the R value on the left.
  expect_identical(as_array(sqrt(2) / x), sqrt(2))
})

test_that("an R value passed as a jit argument reaches f64 exactly", {
  f <- jit(function(t) nv_scalar(-1, dtype = "f64") / t)
  expect_identical(as_array(f(sqrt(2))), -1 / sqrt(2))
})

test_that("constants in the body of an nv_* function are exact", {
  # nv_log2()/nv_log10() divide by `log(2)` / `log(10)` written as R literals.
  x <- nv_array(c(1, 2, 4, 8), dtype = "f64")
  expect_equal(as.vector(nv_log2(x)), log2(c(1, 2, 4, 8)), tolerance = 1e-15)
  y <- nv_array(c(1, 10, 100, 1000), dtype = "f64")
  expect_equal(as.vector(nv_log10(y)), log10(c(1, 10, 100, 1000)), tolerance = 1e-15)
})

test_that("an R array reaches an f64 use site exactly", {
  x <- nv_array(c(1, 1), dtype = "f64")
  expect_identical(as.vector(x * array(c(0.1, sqrt(2)))), c(0.1, sqrt(2)))
})

test_that("an R value is uploaded in its own R category", {
  # A logical can only be handed to the runtime as a logical, an integer as an
  # integer: the upload stays in the value's category and the program converts
  # out of it. Getting this wrong makes the upload itself fail.
  x32 <- nv_array(c(1, 2), dtype = "f32")
  expect_equal(as.vector(nv_mul(x32, TRUE)), c(1, 2))
  expect_equal(as.vector(nv_add(TRUE, nv_array(1:2))), c(2L, 3L))
  expect_equal(as.vector(nv_add(TRUE, nv_scalar(1, dtype = "f64"))), 2)
  # an R logical array as a mask
  expect_equal(as.vector(nv_mul(x32, array(c(TRUE, FALSE)))), c(1, 0))
  # explicit conversions across the category boundary, in both directions
  expect_equal(as.vector(jit(function(x) nv_convert(x, "i32"))(TRUE)), 1L)
  expect_equal(as.vector(jit(function(x) nv_convert(x, "bool"))(1L)), TRUE)
})

test_that("an R double converted to an integer dtype keeps every digit", {
  # The double is uploaded as f64 -- the one float that holds it exactly -- so
  # the conversion sees the value itself rather than an f32 of it.
  f <- jit(function(x) nv_convert(x, "i64"))
  expect_equal(as_array(f(3e9)), 3000000000)
  expect_equal(as_array(f(1e18)), 1e18)
  expect_equal(as_array(f(-3e9)), -3000000000)
  # ... and still truncates toward zero, like converting a typed array does.
  expect_equal(as_array(f(1.9)), 1)
  expect_equal(as_array(f(-1.9)), -1)
})

test_that("a bare R value works as a gradient argument", {
  expect_equal(as_array(jit(gradient(function(v) v * v))(2)[[1L]]), 4)
  expect_equal(as_array(jit(gradient(function(v) v * v))(nv_scalar(2))[[1L]]), 4)
})

test_that("a bare R value works as a subscript", {
  x <- nv_array(c(1, 2, 3), dtype = "f32")
  expect_equal(as.vector(jit(function(a, i) a[i])(x, 2L)), 2)
  expect_equal(as.vector(jit(function(a) a[2])(x)), 2)
})

test_that("nv_array() gives a traced R value a dtype", {
  # This is what the `dtype()` error tells the user to reach for, so it has to
  # work -- and it has to be exact.
  f <- jit(function(v) nv_array(v, dtype = "f64"))
  expect_identical(as_array(f(sqrt(2))), sqrt(2))
  expect_equal(dtype(jit(function(v) nv_scalar(v, dtype = "i64"))(2L)), as_dtype("i64"))
  # Without a dtype it commits to its default.
  expect_equal(dtype(jit(function(v) nv_array(v))(1)), as_dtype("f32"))
  # A value that already has a dtype is not rebuilt this way.
  expect_error(jit(function(v) nv_array(v + 1, dtype = "f64"))(1), "traced value")
})

test_that("dtype_abstract answers for an R value, dtype does not", {
  # The API needs "what would this commit to" without forcing a commitment.
  seen <- NULL
  invisible(jit(function(x) {
    seen <<- dtype_abstract(x)
    x + nv_scalar(1, dtype = "f64")
  })(sqrt(2)))
  expect_equal(seen, as_dtype("f32"))
  expect_equal(dtype_abstract(1.5), as_dtype("f32"))
  expect_equal(dtype_abstract(1L), as_dtype("i32"))
})

test_that("an R value commits to the default dtype when nothing claims it", {
  expect_equal(dtype(jit(function() 1)()), as_dtype("f32"))
  expect_equal(dtype(jit(function() 1L)()), as_dtype("i32"))
  expect_equal(dtype(jit(function() TRUE)()), as_dtype("bool"))
  expect_equal(dtype(jit(identity)(1)), as_dtype("f32"))
  expect_equal(dtype(jit(identity)(array(1:4))), as_dtype("i32"))
})

test_that("an R value takes the dtype it meets, whatever the width", {
  expect_equal(dtype(nv_array(1L, dtype = "i8") + 1), as_dtype("f32"))
  expect_equal(dtype(nv_array(1L, dtype = "i8") + 1L), as_dtype("i8"))
  expect_equal(dtype(nv_array(1, dtype = "f64") + 1L), as_dtype("f64"))
  expect_equal(dtype(nv_array(TRUE) + 1L), as_dtype("i32"))
})

test_that("the compiled program does not depend on the R value", {
  f <- jit(function(x, y) x + y)
  x <- nv_scalar(0, dtype = "f64")
  expect_identical(as_array(f(x, sqrt(2))), sqrt(2))
  # Same key, so this is a cache hit -- and it must still return its own value,
  # not the one the program was compiled with.
  expect_identical(as_array(f(x, pi)), pi)
  expect_equal(cache_size(f), 1L)
})

test_that("one R argument used at two dtypes is exact at the wider one", {
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

test_that("an unused R argument is still an input the call supplies", {
  f <- jit(function(x, y) x + 1)
  expect_identical(as_array(f(nv_scalar(1, dtype = "f64"), 99)), 2)
})

test_that("an anchored literal is embedded as f64, with no convert", {
  graph <- trace_fn(function(x) x / sqrt(2), list(x = nv_aval("f64", integer())))
  src <- repr(stablehlo(graph)[[1L]])
  expect_match(src, "tensor<f64>", fixed = TRUE)
  expect_no_match(src, "convert", fixed = TRUE)
})

test_that("a program without f64 stays free of it", {
  graph <- trace_fn(function(x) x / sqrt(2), list(x = nv_aval("f32", integer())))
  src <- repr(stablehlo(graph)[[1L]])
  expect_no_match(src, "f64", fixed = TRUE)
})

test_that("an R value has no dtype to report", {
  expect_error(jit(function(x) dtype(x))(1), "no data type of its own")
  # The message names the way out.
  expect_error(jit(function(x) dtype(x))(1), "nv_array")
  expect_error(jit(function(x) x + dtype(x))(array(1:4)), "no data type of its own")
  # `nv_*_like` derives the result's dtype from its argument, so it errors too.
  expect_error(jit(function(x) nv_fill_like(x, 3))(1), "no data type of its own")
  # Giving it a dtype answers the question.
  expect_equal(dtype(jit(function(x) nv_fill_like(x, 3))(nv_scalar(1))), as_dtype("f32"))
})

test_that("an R value does have a shape", {
  seen <- list()
  f <- jit(function(x) {
    seen[["shape"]] <<- shape(x)
    seen[["naxes"]] <<- naxes(x)
    x + 1
  })
  invisible(f(array(1:6, c(2, 3))))
  expect_equal(seen$shape, c(2L, 3L))
  expect_equal(seen$naxes, 2L)

  seen <- list()
  invisible(f(1))
  expect_equal(seen$shape, integer())
  expect_equal(seen$naxes, 0L)
})

test_that("nv_convert builds an R value at the target dtype directly", {
  expect_identical(as_array(jit(function() nv_convert(sqrt(2), "f64"))()), sqrt(2))
  expect_identical(as_array(nv_convert(sqrt(2), "f64")), sqrt(2))
  # Converting to an integer dtype truncates, as it does for a typed array.
  expect_equal(as_array(nv_convert(1.9, "i32")), 1L)
  expect_equal(as_array(nv_convert(-1.9, "i32")), -1L)
})

test_that("assigning an R value into an array uses the array's dtype", {
  x <- nv_array(c(1, 2, 3), dtype = "f64")
  x[2] <- 0.1
  expect_identical(as.vector(x), c(1, 0.1, 3))
  # What may be assigned is still decided by the dtype the R value would take.
  y <- nv_array(1:3, dtype = "i32")
  expect_error(y[2] <- 1.5, "not promotable")
})

test_that("a literal that only meets other literals takes the default", {
  # The f64 arrives on `y`, one step after `x` has already committed. This is
  # the documented limit of committing per operation.
  f <- jit(function(x) {
    y <- x * 2
    y + nv_scalar(1, dtype = "f64")
  })
  out <- f(sqrt(2))
  expect_equal(dtype(out), as_dtype("f64"))
  expect_false(isTRUE(all.equal(as_array(out), 2 * sqrt(2) + 1, tolerance = 1e-15)))
})

test_that("RDataArray reports what the R value can answer", {
  x <- RDataArray(1.5, integer())
  expect_equal(shape(x), integer())
  expect_error(dtype(x), "no data type of its own")

  y <- RDataArray(array(1:6, c(2, 3)), c(2L, 3L))
  expect_equal(shape(y), c(2L, 3L))
  expect_equal(naxes(y), 2L)
})

test_that("quickr: an R value reaches an f64 use site exactly", {
  skip_if_no_quickr()
  local_backend("quickr")
  x <- nv_scalar(1, dtype = "f64")
  expect_identical(as_array(jit(function(x) x / sqrt(2))(x)), 1 / sqrt(2))
  expect_identical(as_array(jit(function(x, y) x / y)(x, sqrt(2))), 1 / sqrt(2))
})

test_that("quickr: an R argument is coerced to the dtype the program takes", {
  skip_if_no_quickr()
  local_backend("quickr")
  # The leaf arrives as an R integer but the program consumes it as f64.
  f <- jit(function(x, y) x + y)
  expect_identical(as_array(f(nv_scalar(1, dtype = "f64"), 2L)), 3)
})
