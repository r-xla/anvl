test_that("error message when using different platforms", {
  skip_if(!is_cuda())
  f <- jit(\(x, y) x + y)
  x <- nv_array(1, device = "cpu")
  y <- nv_array(1, device = "cuda")
  # pjrt's dispatcher validates device agreement natively.
  expect_error(f(x, y), "lives on a different device")
})

# A backend-specific argument is checked when the implementation for the
# backend in force is built, on the first call.
test_that("donate: must be formal args of f", {
  expect_error(jit(function(x) x, donate = "y")(nv_array(1)), "subset of")
})

test_that("donate: cannot also be static", {
  expect_error(jit(function(x, y) x, donate = "x", static = "x")(nv_array(1), nv_array(2)), "donate.*static")
})

test_that("donate: no aliasing with type mismatch", {
  skip_if(!is_cpu()) # might get a segfault on other platforms
  f <- jit(function(x) x, device = "cpu", donate = "x")
  x <- nv_array(1)
  out <- f(x)
  expect_error(capture.output(x), "called on deleted or donated buffer")
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
