# jit() driving pjrt's native dispatcher, end to end.
#
# These live here rather than in pjrt because they are an *integration* check:
# that anvl's real compile callback returns what pjrt's engine expects, that a
# real AnvlArray carries $data/$backend/$device where the engine looks for them,
# and that the wrapped outputs come back as anvl arrays. pjrt tests its
# dispatcher against hand-written fixtures; only anvl can test that the two
# actually agree.

skip_if_no_jit <- function() {
  testthat::skip_if_not(pjrt::plugins_downloaded())
}

# `jit_dispatcher()` is anvl's own accessor for the dispatcher of the backend in
# force; a jitted function holds one implementation per backend.
jit_size <- function(f) pjrt::dispatcher_size(jit_dispatcher(f))

arr_of <- function(res) as.numeric(tengen::as_array(res))

test_that("jit() dispatches, caches, and returns wrapped arrays", {
  skip_if_no_jit()
  f <- jit(function(x, y) x + y)
  x <- nv_array(c(1, 2, 3), dtype = "f32")
  y <- nv_array(c(10, 20, 30), dtype = "f32")

  r1 <- f(x, y)
  # The result is a fully wrapped array: the dispatcher built it natively.
  expect_s3_class(r1, "AnvlArray")
  expect_identical(as.character(tengen::dtype(r1)), "f32")
  expect_identical(tengen::shape(r1), 3L)
  expect_s3_class(r1$data, "PJRTBuffer")
  expect_s3_class(tengen::device(r1), "PJRTDevice")
  expect_identical(r1$backend, "pjrt")
  expect_equal(arr_of(r1), c(11, 22, 33))

  # A second call of the same signature is a cache hit...
  expect_equal(arr_of(f(x, y)), c(11, 22, 33))
  d <- jit_dispatcher(f)
  expect_s3_class(d, "Dispatcher")
  expect_equal(pjrt::dispatcher_size(d), 1L)

  # ...an output feeds straight back in as an input, without re-compiling...
  expect_equal(arr_of(f(r1, y)), c(21, 42, 63))
  expect_equal(pjrt::dispatcher_size(d), 1L)

  # ...and a new shape is a new cache entry.
  invisible(f(nv_array(1, dtype = "f32"), nv_array(2, dtype = "f32")))
  expect_equal(pjrt::dispatcher_size(d), 2L)

  # GC-correct: many dispatches with periodic gc(), then teardown.
  for (i in 1:300) {
    r <- f(x, y)
    if (i %% 100 == 0) {
      gc()
    }
    expect_equal(arr_of(r), c(11, 22, 33))
  }
})

test_that("jit() preserves nested output structure and names", {
  skip_if_no_jit()
  f <- jit(function(x) list(sum = x + x, nested = list(sq = x * x)))
  res <- f(nv_array(c(2, 3), dtype = "f32"))
  expect_named(res, c("sum", "nested"))
  expect_named(res$nested, "sq")
  expect_equal(arr_of(res$sum), c(4, 6))
  expect_equal(arr_of(res$nested$sq), c(4, 9))
})

test_that("jit() with static args compiles per static value", {
  skip_if_no_jit()
  f <- jit(function(x, flag) if (flag) x + 1 else x * 2, static = "flag")
  x <- nv_array(3, dtype = "f32")
  expect_equal(arr_of(f(x, TRUE)), 4)
  expect_equal(arr_of(f(x, FALSE)), 6)
  expect_equal(arr_of(f(x, TRUE)), 4) # hit
  expect_equal(jit_size(f), 2L)
})

test_that("a jitted call with no dynamic input dispatches on its statics alone", {
  skip_if_no_jit()
  # Zero dynamic leaves: the whole call is the static `n`, and the entry's
  # device comes from the compile callback rather than from an input.
  f <- jit(function(n) nv_eye(n), static = "n")
  expect_equal(tengen::as_array(f(2L)), diag(2))
  expect_equal(tengen::as_array(f(2L)), diag(2))
  expect_equal(jit_size(f), 1L)
})

test_that("jit() uploads bare R literals and arrays", {
  skip_if_no_jit()
  f <- jit(function(x, y) x + y)
  x <- nv_array(c(1, 2), dtype = "f32")

  # A bare double literal is uploaded as a rank-0 f32 buffer per call; the
  # signature does not change, so the second call is a cache hit.
  expect_equal(arr_of(f(x, 5)), c(6, 7))
  expect_equal(arr_of(f(x, 50)), c(51, 52))
  expect_equal(jit_size(f), 1L)

  # kArray and kRData are *different* key material: bare R data has no dtype of
  # its own, so it cannot share an entry with an array that has one -- the two
  # compile to different programs, one taking an f32 input and one uploading the
  # R value at whatever dtype the trace decided.
  expect_equal(arr_of(f(x, nv_scalar(5, dtype = "f32"))), c(6, 7))
  expect_equal(jit_size(f), 2L)

  # A different R storage type is a different key again: `3L` is an R integer,
  # keyed apart from an R double even though both end up at f32 here.
  expect_equal(arr_of(f(x, 3L)), c(4, 5))
  expect_equal(jit_size(f), 3L)

  # An R array leaf uploads column-major, like pjrt_buffer().
  g <- jit(function(x) x)
  m <- matrix(c(1, 2, 3, 4), nrow = 2)
  expect_equal(tengen::as_array(g(m)), m)
})

test_that("every dtype is its own cache entry", {
  skip_if_no_jit()
  # Every dtype an AnvlDtype names -- which is every dtype tengen can build.
  dtypes <- c("bool", "i8", "i16", "i32", "i64", "ui8", "ui16", "ui32", "ui64", "f32", "f64")
  f <- jit(function(x) x)
  for (dt in dtypes) {
    invisible(f(nv_array(c(1, 2), dtype = dt)))
  }
  expect_equal(jit_size(f), length(dtypes))

  # Same dtype and shape, different values -> cache hit.
  g <- jit(function(x) x)
  invisible(g(nv_array(c(1, 2), dtype = "f64")))
  invisible(g(nv_array(c(7, 7), dtype = "f64")))
  expect_equal(jit_size(g), 1L)
})

# What follows is how the cache key treats real R values as static arguments.

test_that("invalid jit() inputs are rejected natively, naming the argument", {
  skip_if_no_jit()
  f <- jit(function(x, y) x + y)
  x <- nv_array(c(1, 2), dtype = "f32")
  expect_error(f(x, "nope"), "invalid input `y`.*<character> of length 1")
  expect_error(f(x, c(1, 2, 3)), "invalid input `y`.*<numeric> of length 3")
  expect_equal(jit_size(f), 0L) # rejected before any compile

  # A static argument must not be an AnvlArray: it would key the cache on its
  # contents, and be traced as an input execution never supplies.
  g <- jit(function(x, s) x + 1, static = "s")
  expect_error(g(x, x), "invalid static input `s`.*must not be an AnvlArray")
  expect_equal(jit_size(g), 0L)
})

test_that("jit() rejects inputs spread across devices, naming the input", {
  skip_if_no_jit()
  skip_if(length(pjrt::devices(pjrt::pjrt_client("cpu"))) < 2L, "needs a second cpu device")
  f <- jit(function(x, y) x + y)
  x0 <- nv_array(c(1, 2), dtype = "f32", device = "cpu:0")
  y1 <- nv_array(c(3, 4), dtype = "f32", device = "cpu:1")

  # Without a fixed target device the first array's device is the call's, and a
  # conflicting input is an error -- caught natively, before the cache is
  # probed, so nothing is compiled.
  expect_error(f(x0, y1), "invalid input `y`.*different device")
  expect_equal(jit_size(f), 0L)
})

test_that("jit(device = ) fixes the entry's device and moves inputs to it", {
  skip_if_no_jit()
  skip_if(length(pjrt::devices(pjrt::pjrt_client("cpu"))) < 2L, "needs a second cpu device")
  f <- jit(function(x) x + 1, device = "cpu:0")
  x0 <- nv_array(c(1, 2), dtype = "f32", device = "cpu:0")
  res <- f(x0)
  expect_equal(arr_of(res), c(2, 3))
  # Devices are interned, so the wrapped output carries the very object.
  expect_identical(tengen::device(res), pjrt::pjrt_device("cpu:0"))

  # An input on another device is copied to the target rather than rejected,
  # and the device is not part of the key: one entry serves both.
  y1 <- nv_array(c(3, 4), dtype = "f32", device = "cpu:1")
  expect_equal(arr_of(f(y1)), c(4, 5))
  expect_equal(jit_size(f), 1L)
})

test_that("a jitted function with no array inputs keys on the default device", {
  skip_if_no_jit()
  f <- jit(function(n) n + 1)
  expect_equal(arr_of(f(41)), 42)
  expect_equal(arr_of(f(41)), 42)
  expect_equal(jit_size(f), 1L)
})

test_that("the quickr backend dispatches through the closure engine", {
  skip_if_no_quickr()
  with_backend("quickr", {
    f <- jit(function(x, y) x + y)
    x <- nv_array(c(1, 2), dtype = "f64")
    y <- nv_array(c(10, 20), dtype = "f64")
    r1 <- f(x, y)
    expect_s3_class(r1, "AnvlArray")
    expect_identical(r1$backend, "quickr")
    expect_equal(arr_of(r1), c(11, 22))
    invisible(f(x, y))
    expect_equal(jit_size(f), 1L)
  })
})
