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
  # `i64` comes back as a bit64::integer64; compare its digits, which is the
  # only comparison that stays exact past 2^53.
  f <- function(x) format(as_array(jit(function(v) nv_convert(v, "i64"))(x)))
  expect_equal(f(3e9), "3000000000")
  expect_equal(f(1e18), "1000000000000000000")
  expect_equal(f(-3e9), "-3000000000")
  # ... and still truncates toward zero, like converting a typed array does.
  expect_equal(f(1.9), "1")
  expect_equal(f(-1.9), "-1")
})

test_that("a bare R value works as a gradient argument", {
  expect_equal(as_array(jit(gradient(function(v) v * v))(2)[[1L]]), 4)
  expect_equal(as_array(jit(gradient(function(v) v * v))(nv_scalar(2))[[1L]]), 4)
  # ... and as a literal in the body of the function being differentiated
  expect_equal(jit(gradient(function(x) x * 2, wrt = "x"))(nv_scalar(1)), list(x = nv_scalar(2)))
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

test_that("peek_dtype answers for an R value, dtype does not", {
  # The API needs "what would this commit to" without forcing a commitment.
  seen <- NULL
  invisible(jit(function(x) {
    seen <<- peek_dtype(x)
    x + nv_scalar(1, dtype = "f64")
  })(sqrt(2)))
  expect_equal(seen, as_dtype("f32"))
  expect_equal(peek_dtype(1.5), as_dtype("f32"))
  expect_equal(peek_dtype(1L), as_dtype("i32"))
})

test_that("an R value commits to the default dtype when nothing claims it", {
  expect_equal(dtype(jit(function() 1)()), as_dtype("f32"))
  expect_equal(dtype(jit(function() 1L)()), as_dtype("i32"))
  expect_equal(dtype(jit(function() TRUE)()), as_dtype("bool"))
  expect_equal(dtype(jit(identity)(1)), as_dtype("f32"))
  expect_equal(dtype(jit(identity)(array(1:4))), as_dtype("i32"))
  # ... including a value that only ever meets other R values
  expect_equal(jit(function() nv_mul(2, 3))(), nv_scalar(6, dtype = "f32"))
  expect_equal(jit(function() nv_mul(2, 3L))(), nv_scalar(6, dtype = "f32"))
  expect_equal(jit(function(x) x + 1)(1), nv_scalar(2, dtype = "f32"))
})

test_that("an R value takes the dtype it meets, whatever the width", {
  expect_equal(dtype(nv_array(1L, dtype = "i8") + 1), as_dtype("f32"))
  expect_equal(dtype(nv_array(1L, dtype = "i8") + 1L), as_dtype("i8"))
  expect_equal(dtype(nv_array(1, dtype = "f64") + 1L), as_dtype("f64"))
  expect_equal(dtype(nv_array(TRUE) + 1L), as_dtype("i32"))
  # narrower than the value's own default, and on either side of the operator
  expect_equal(jit(function(x) x * 2L)(nv_scalar(1, dtype = "i16")), nv_scalar(2L, dtype = "i16"))
  expect_equal(jit(function(x) 2 + x)(nv_scalar(1)), nv_scalar(3))
  expect_equal(jit(function(x) nv_mul(2, x))(nv_scalar(3, dtype = "f64")), nv_scalar(6, dtype = "f64")) # nolint
  expect_equal(jit(function(x) x == TRUE)(nv_scalar(FALSE)), nv_scalar(FALSE))
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
  # What may be assigned is still decided by the dtype the R value would take:
  # an R double has no place in an integer array, whatever its value.
  y <- nv_array(1:3, dtype = "i32")
  expect_error(y[2] <- 1.5, "R double")
  expect_error(y[2] <- 1, "R double")
  y[2] <- 5L
  expect_identical(as.vector(y), c(1L, 5L, 3L))
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

test_that("a bare R value answers the extractors like an RDataArray", {
  # Eagerly the R value *is* the uncommitted value, so it has to answer the way
  # the boxed one does under jit().
  expect_equal(shape(1.5), integer())
  expect_equal(naxes(1.5), 0L)
  expect_equal(shape(TRUE), integer())
  expect_equal(shape(array(1:6, c(2, 3))), c(2L, 3L))
  expect_error(dtype(1.5), "no data type of its own")
  expect_error(dtype(1L), "no data type of its own")
  expect_error(dtype(TRUE), "no data type of its own")
  expect_error(dtype(array(1:6, c(2, 3))), "no data type of its own")
  expect_equal(peek_dtype(1.5), as_dtype("f32"))
  # A vector that is not an anvl value at all says so rather than lying.
  expect_error(shape(c(1, 2, 3)), "undefined for a length-3")
})

test_that("the graph records the dtype each R input is uploaded at", {
  graph <- trace_fn(
    function(x, y) x + y,
    list(x = nv_aval("f64", 2L), y = nv_aval("double", integer()))
  )
  # It lives on the input's own aval, not beside it.
  expect_s3_class(graph$inputs[[1L]]$aval, "AbstractArray")
  expect_false(is_rdata_input(graph$inputs[[1L]]$aval))
  expect_s3_class(graph$inputs[[2L]]$aval, "RDataInput")
  expect_equal(dtype(graph$inputs[[2L]]$aval), as_dtype("f64"))
  expect_equal(graph$inputs[[2L]]$aval$r_type, "double")
  # ... and the vector the backends read is derived from it.
  expect_equal(graph_input_dtypes(graph), c(NA, "f64"))
  # No R input at all means nothing to upload, and the backends skip the step.
  plain <- trace_fn(function(x) x + 1, list(x = nv_aval("f64", 2L)))
  expect_null(graph_input_dtypes(plain))
})

test_that("nv_pad builds the padding value at the array's dtype", {
  for (dt in c("f64", "f32", "i8", "i32")) {
    x <- nv_array(c(1L, 2L), dtype = dt)
    # An integer literal reaches any of them; a double only the floats, since
    # it would have to leave its category to become an integer.
    expect_equal(dtype(nv_pad(x, 0L, 1L, 1L)), as_dtype(dt), info = dt)
    if (is_dtype_float(as_dtype(dt))) {
      expect_equal(dtype(nv_pad(x, 0, 1L, 1L)), as_dtype(dt), info = dt)
    } else {
      expect_error(nv_pad(x, 0, 1L, 1L), "R double", info = dt)
    }
  }
  x <- nv_array(c(1, 2), dtype = "f64")
  expect_identical(as.vector(nv_pad(x, sqrt(2), 1L, 0L)), c(sqrt(2), 1, 2))
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


# --- What a jitted call accepts as a bare R argument -------------------------

test_that("a jitted call converts an R argument of any R type", {
  f <- jit(identity)
  expect_equal(f(1), nv_scalar(1))
  expect_equal(f(1L), nv_scalar(1L))
  # A logical is the one R type that names its dtype: there is nothing for
  # `TRUE` to become other than `bool`.
  expect_equal(f(TRUE), nv_scalar(TRUE))
  expect_equal(f(matrix(1:4, 2, 2)), nv_matrix(1:4, nrow = 2, ncol = 2))
  out <- f(array(1:24, dim = c(2, 3, 4)))
  expect_equal(dtype(out), as_dtype("i32"))
  expect_equal(shape(out), c(2L, 3L, 4L))
})

test_that("R leaves of a nested argument are converted too", {
  f <- jit(function(pair) pair[[1]] + pair[[2]])
  out <- f(list(1, 2))
  expect_equal(dtype(out), as_dtype("f32"))
  expect_equal(as_array(out), 3)
})

test_that("a static argument is not converted", {
  f <- jit(function(x, flag) if (flag) x + 1 else x * 2, static = "flag")
  expect_equal(as_array(f(nv_scalar(3), TRUE)), 4)
  expect_equal(as_array(f(3, FALSE)), 6)
})

test_that("a traced value passed to an inner jit is left alone", {
  inner <- jit(function(x) x + 1)
  outer <- jit(function(x) inner(x))
  expect_equal(as_array(outer(nv_scalar(1))), 2)
})

test_that("jit: bare vector without dim errors", {
  f <- jit(function(x) x)
  expect_snapshot(f(c(1, 2, 3)), error = TRUE)
})

test_that("jit: non-array/non-scalar leaves (e.g. character) error", {
  f <- jit(function(x) x)
  expect_snapshot(f("hello"), error = TRUE)
})

test_that("jit: error shows path for nested list element", {
  f <- jit(function(l) l[[1]])
  expect_snapshot(f(list(list(a = "abc"))), error = TRUE)
})

test_that("jit: error shows path for unnamed nested element", {
  f <- jit(function(pair) pair[[1]])
  expect_snapshot(f(list("bad", nv_scalar(1))), error = TRUE)
})

test_that("quickr: an R argument is converted the same way", {
  skip_if_no_quickr()
  local_backend("quickr")
  f <- jit(identity)
  expect_equal(f(1), nv_scalar(1))
  expect_equal(f(matrix(1:4, 2, 2)), nv_matrix(1:4, nrow = 2, ncol = 2))
  g <- jit(function(pair) pair[[1]] + pair[[2]])
  expect_equal(g(list(nv_scalar(1L), 2L)), nv_scalar(3L))
})

test_that("quickr: bare vector errors", {
  skip_if_no_quickr()
  local_backend("quickr")
  f <- jit(function(x) x)
  expect_snapshot(f(c(1, 2, 3)), error = TRUE)
})

test_that("an RDataArray does not print the R data it carries", {
  # The data can be a whole array, and what matters about the value is what it
  # is, not what it holds.
  expect_identical(format(RDataArray(1.5, integer())), "RDataArray(double, ())")
  expect_identical(
    format(RDataArray(array(1:6, c(2, 3)), c(2L, 3L))),
    "RDataArray(integer, (2,3))"
  )
  # An argument of a jitted function has no data to print in the first place.
  expect_identical(
    format(RDataArray(NULL, integer(), "logical")),
    "RDataArray(logical, ())"
  )
  expect_identical(repr(RDataArray(1.5, c(2L, 3L))), "double[2x3]")
})

test_that("an RDataInput names both data types it stands between", {
  x <- RDataInput("f64", c(2L, 3L), "double")
  expect_identical(format(x), "RDataInput(f64, double, (2,3))")
  expect_identical(repr(x), "f64[2x3]<-double")
})

test_that("nv_rnorm takes its data type from mean and sd when none is named", {
  state <- nv_rng_state(42L)
  # Bare R values have no data type, so the sample falls back to the default
  # float rather than to whatever R stores its numbers as.
  expect_equal(dtype(nv_rnorm(4L, state, mean = 0, sd = 1)[[2L]]), as_dtype("f32"))
  # A real array names it.
  expect_equal(
    dtype(nv_rnorm(4L, state, mean = nv_scalar(0, dtype = "f64"))[[2L]]),
    as_dtype("f64")
  )
  expect_equal(
    dtype(nv_rnorm(4L, state, sd = nv_scalar(1, dtype = "f64"))[[2L]]),
    as_dtype("f64")
  )
  # An integer array cannot name one, so the default float stands.
  expect_equal(dtype(nv_rnorm(4L, state, mean = nv_scalar(0L))[[2L]]), as_dtype("f32"))
  # ... and an explicit `dtype` still wins.
  expect_equal(
    dtype(nv_rnorm(4L, state, dtype = "f64", mean = 0)[[2L]]),
    as_dtype("f64")
  )
  # An f64 mean cannot be narrowed to an f32 sample.
  expect_error(
    nv_rnorm(4L, state, dtype = "f32", mean = nv_scalar(0, dtype = "f64")),
    "not promotable"
  )
})

test_that("an out-of-category R leaf is built in each sub-graph that uses it", {
  # The convert is recorded in whatever descriptor is being traced, so a memo
  # entry from one branch must not be handed to the other -- the second branch
  # would reference a value only the first one computes.
  f <- jit(function(a, x) {
    prim_if(nv_scalar(TRUE), function() nv_add(x, a), function() nv_add(x, a))
  })
  expect_identical(as_array(f(3L, nv_scalar(2, dtype = "f64"))), 5)
  g <- jit(function(a, x) {
    nv_while(list(i = x), function(i) i < nv_add(x, a), function(i) list(i = nv_add(i, a)))
  })
  expect_identical(as_array(g(3L, nv_scalar(2, dtype = "f64"))$i), 5)
})

test_that("a negative R integer reaches an unsigned data type through a convert", {
  # An R integer is signed, so it is built at i32/i64 and converted: writing it
  # straight into the IR would not even be valid StableHLO, and the answer has
  # to be the same eagerly as under jit().
  x <- nv_array(c(1L, 2L), dtype = "ui32")
  expect_equal(as_array(nv_add(x, -1L)), as_array(jit(function(x) nv_add(x, -1L))(x)))
  expect_equal(as.character(as_array(nv_add(x, -1L))), c("0", "1"))
  expect_equal(as.character(as_array(nv_add(x, 1L))), c("2", "3"))
  graph <- trace_fn(function(x) nv_add(x, -1L), list(x = nv_aval("ui32", 2L)))
  src <- repr(stablehlo(graph)[[1L]])
  expect_match(src, "dense<-1> : tensor<i32>", fixed = TRUE)
  expect_match(src, "stablehlo.convert", fixed = TRUE)
})

test_that("nv_solve() and nv_triangular_solve() let an R value yield", {
  a <- nv_array(matrix(c(2, 0, 0, 2), nrow = 2), dtype = "f64")
  out <- nv_solve(a, matrix(c(2, 4), ncol = 1L))
  expect_equal(dtype(out), as_dtype("f64"))
  expect_equal(as.vector(out), c(1, 2))
  # ... and two typed arrays that disagree are still rejected, rather than one
  # of them being widened.
  expect_error(nv_solve(a, nv_array(matrix(c(2, 4), ncol = 1L), dtype = "f32")))
  out <- nv_triangular_solve(a, matrix(c(2, 4), ncol = 1L))
  expect_equal(dtype(out), as_dtype("f64"))
})

test_that("an R vector that is not arrayish has no abstract value", {
  expect_error(to_abstract(c(1, 2, 3)), "undefined for a length-3")
  expect_equal(shape(to_abstract(array(1:6, c(2, 3)))), c(2L, 3L))
})

test_that("an R argument uploads at the narrowest data type that holds every use site", {
  dbl <- RDataArray(NULL, integer(), "double")
  # `f16` and `bf16` are both 16 bits and neither holds the other, so the upload
  # has to widen -- to `f32`, which holds both, and not to the value's natural
  # `f64`: a program with no `f64` in it must not acquire one here.
  expect_equal(resolve_upload_dtype(dbl, c("f16", "bf16")), "f32")
  # Where one of them does hold the others, that one is used unchanged.
  expect_equal(resolve_upload_dtype(dbl, c("f32", "f64")), "f64")
  expect_equal(resolve_upload_dtype(dbl, "f16"), "f16")
  # A value the body never used commits to its default.
  expect_equal(resolve_upload_dtype(dbl, character()), "f32")
  expect_equal(resolve_upload_dtype(RDataArray(NULL, integer(), "integer"), c("i32", "i64")), "i64")
})
