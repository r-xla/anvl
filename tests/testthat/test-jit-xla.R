test_that("error message when using different platforms", {
  skip_if(!is_cuda())
  f <- jit(\(x, y) x + y)
  x <- nv_array(1, device = "cpu")
  y <- nv_array(1, device = "cuda")
  expect_error(f(x, y), "on unexpected device")
})

test_that("donate: must be formal args of f", {
  expect_error(jit(function(x) x, donate = "y"), "subset of")
})

test_that("donate: cannot also be static", {
  expect_error(jit(function(x, y) x, donate = "x", static = "x"), "donate.*static")
})

test_that("donate: no aliasing with type mismatch", {
  skip_if(!is_cpu()) # might get a segfault on other platforms
  f <- jit(function(x) x, device = "cpu", donate = "x")
  x <- nv_array(1)
  out <- f(x)
  expect_error(capture.output(x), "called on deleted or donated buffer")
})

test_that("xla: basic test", {
  f_add <- function(x, y) x + y
  args <- list(x = nv_aval("f32", c()), y = nv_aval("f32", c()))
  f_compiled <- xla(f_add, args = args)
  result <- f_compiled(nv_scalar(1, dtype = "f32"), nv_scalar(2, dtype = "f32"))
  expect_equal(result, nv_scalar(3, dtype = "f32"))
})

describe("CPU output donation", {
  it("appends a phantom input per non-aliased output on CPU", {
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

  it("skips phantoms on non-CPU platforms even with donate_unaliased_outputs", {
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

  it("skips phantoms on CPU when donate_unaliased_outputs = FALSE", {
    skip_if(!is_cpu())
    graph <- trace_fn(
      function(x) x + 1,
      list(x = nv_aval("f32", shape = c(4L)))
    )
    out <- stablehlo(graph, platform = "cpu")
    expect_length(out[[3L]], 0L)
  })

  it("does not add a phantom for a user-donated input (output already aliased)", {
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
})
