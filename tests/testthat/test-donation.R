# Tests for anvl's CPU output-donation pipeline.
#
# On CPU, anvl asks `stablehlo()` to append a phantom donated input for
# every output that isn't already aliased to a user-`donate`d input.
# The backend allocates a `pjrt_empty()` buffer per phantom at execute
# time, and `pjrt::pjrt_execute()` migrates the RAWSXP keepalive onto
# each aliased output's XPtr. The net effect is that jitted-function
# output bytes live in R-owned RAWSXPs whose lifetime is bound to the
# output XPtr.

is_cpu <- function() Sys.getenv("PJRT_PLATFORM", "cpu") == "cpu"

# Read the protected SEXP off an external pointer for test introspection.
xptr_prot <- local({
  fn <- NULL
  function(x) {
    if (is.null(fn)) {
      fn <<- Rcpp::cppFunction(
        'SEXP xptr_prot_impl(SEXP x) { return R_ExternalPtrProtected(x); }'
      )
    }
    fn(x)
  }
})

test_that("CPU stablehlo lowering appends a phantom input per non-aliased output", {
  skip_if(!is_cpu())
  graph <- trace_fn(
    function(x) x + 1,
    list(x = nv_aval("f32", shape = c(4L)))
  )
  out <- stablehlo(
    graph,
    donate_unaliased_outputs = TRUE,
    platform = "cpu"
  )
  phantom_specs <- out[[3L]]
  expect_length(phantom_specs, 1L)
  expect_equal(phantom_specs[[1L]]$shape, 4L)
  expect_equal(as.character(phantom_specs[[1L]]$dtype), "f32")
})

test_that("non-CPU lowering skips phantoms even with donate_unaliased_outputs", {
  graph <- trace_fn(
    function(x) x + 1,
    list(x = nv_aval("f32", shape = c(4L)))
  )
  out <- stablehlo(
    graph,
    donate_unaliased_outputs = TRUE,
    platform = "cuda"
  )
  expect_length(out[[3L]], 0L)
})

test_that("CPU lowering also skips phantoms when donate_unaliased_outputs = FALSE", {
  skip_if(!is_cpu())
  graph <- trace_fn(
    function(x) x + 1,
    list(x = nv_aval("f32", shape = c(4L)))
  )
  out <- stablehlo(graph, platform = "cpu")
  expect_length(out[[3L]], 0L)
})

test_that("user-donated input does not get a phantom (output already aliased)", {
  skip_if(!is_cpu())
  graph <- trace_fn(
    function(x) x * 2,
    list(x = nv_aval("f32", shape = c(4L)))
  )
  out <- stablehlo(
    graph,
    donate = "x",
    donate_unaliased_outputs = TRUE,
    platform = "cpu"
  )
  # Output 0 is aliased to user input 0 (after constants), so no phantom
  # is needed.
  expect_length(out[[3L]], 0L)
})

test_that("jit output buffers are backed by an R-owned RAWSXP on CPU", {
  skip_if(!is_cpu())
  f <- jit(function(x) x + 1)
  x <- nv_array(c(1, 2, 3, 4), dtype = "f32")
  y <- f(x)

  prot <- xptr_prot(y$data)
  expect_true(is.raw(prot))
  # 4 elements x 4 bytes (f32) = 16
  expect_equal(length(prot), 16L)
  expect_equal(as.numeric(as_array(y)), c(2, 3, 4, 5), tolerance = 1e-6)
})

test_that("output bytes survive GC pressure and stay attached to the AnvlArray", {
  skip_if(!is_cpu())
  f <- jit(function(x) x * 2)
  x <- nv_array(seq(1, 1024 * 256), dtype = "f32")
  y <- f(x)
  rm(x)
  for (i in 1:5) {
    invisible(rnorm(1e5))
    gc(full = TRUE)
  }
  vals <- as.numeric(as_array(y))
  expect_equal(vals[1L], 2, tolerance = 1e-6)
  expect_equal(vals[length(vals)], 2 * 1024 * 256, tolerance = 1e-6)
})

test_that("multi-output jit attaches a RAWSXP to every output", {
  skip_if(!is_cpu())
  f <- jit(function(x) list(a = x + 1, b = x * 3))
  x <- nv_array(c(1, 2, 3, 4), dtype = "f32")
  out <- f(x)
  for (nm in c("a", "b")) {
    p <- xptr_prot(out[[nm]]$data)
    expect_true(is.raw(p), info = nm)
    expect_equal(length(p), 16L, info = nm)
  }
})

test_that("user-donated input transfers its RAWSXP to the aliased output", {
  skip_if(!is_cpu())
  f <- jit(function(x) x * 2, donate = "x")
  x <- nv_array(c(10, 20, 30, 40), dtype = "f32")
  x_prot <- xptr_prot(x$data)
  expect_true(is.raw(x_prot))

  y <- f(x)
  # The output XPtr now holds the donated input's RAWSXP; the input's
  # prot slot has been cleared by pjrt.
  expect_identical(xptr_prot(y$data), x_prot)
  expect_null(xptr_prot(x$data))

  rm(x)
  for (i in 1:3) {
    invisible(rnorm(1e5))
    gc(full = TRUE)
  }
  expect_equal(as.numeric(as_array(y)), c(20, 40, 60, 80), tolerance = 1e-6)
})
