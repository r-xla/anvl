test_that("graph_to_quickr_function requires {quickr}", {
  graph <- trace_fn(
    function(x) x + x,
    list(x = nv_scalar(1.0, dtype = "f64"))
  )

  if (!requireNamespace("quickr", quietly = TRUE)) {
    expect_error(graph_to_quickr_function(graph), "quickr", fixed = FALSE)
    return()
  }

  f <- graph_to_quickr_function(graph)
  expect_equal(as_array(f(2)), 4)
})

describe("the quickr backend", {
  it("rejects a float default it cannot represent", {
    skip_if_no_quickr()
    local_backend("quickr")
    # quickr has no single precision, so `f32` cannot be honoured; it is an
    # error rather than a double labelled `f32`, which is the mislabelling the
    # data type system exists to prevent.
    with_default_dtypes(c(float = "f32"), {
      expect_error(nv_array(1.5), "quickr")
      # A constant of a trace is captured backend-agnostically, so this one is
      # only caught where the program is lowered.
      expect_error(jit(function() nv_array(1.5))(), "quickr")
    })
    expect_error(nv_array(1.5, dtype = "f32"), "quickr")
    expect_error(jit(function(x) nv_convert(x, "f32"))(nv_array(1, dtype = "f64")), "quickr")
  })

  it("commits an R double to f64 everywhere", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_equal(dtype(nv_array(1.5)), as_dtype("f64"))
    expect_equal(peek_dtype(1.5), as_dtype("f64"))
    expect_equal(dtype(jit(function() 1.5)()), as_dtype("f64"))
    expect_equal(dtype(jit(function(x) x + 1.5)(nv_array(1L))), as_dtype("f64"))
    expect_equal(dtype(jit(function(x) x)(1.5)), as_dtype("f64"))
    expect_equal(dtype(nv_array(1L)), as_dtype("i32"))
  })
})

describe("an override quickr cannot represent", {
  it("fails where the data is allocated, not where it is set", {
    skip_if_no_quickr()
    # The option is taken as set (see `test-default-dtypes.R`); quickr has no
    # `i64`, whether the entry names the backend or not.
    local_default_dtypes(c(int = "i64"), backend = "quickr")
    expect_error(with_backend("quickr", nv_array(1L)), "quickr")
    withr::local_options(anvl.default_dtypes = c(int = "i64"))
    expect_error(with_backend("quickr", nv_array(1L)), "quickr")
  })
})
