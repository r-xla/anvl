# The lowering itself is tested on the r backend, which runs the same R function
# uncompiled (test-graph-to-quickr.R, test-primitives-r.R). These tests check
# that quickr compiles it and that jit() runs the result.

describe("quickr_device()", {
  it("is the only device of the quickr backend", {
    dev <- quickr_device("cpu")
    expect_s3_class(dev, "QuickrDevice")
    expect_equal(format(dev), "QuickrDevice(cpu)")
    expect_equal(as.character(dev), "cpu")
    expect_true(dev == quickr_device())
  })
})

describe("the quickr backend", {
  it("compiles a jitted function and returns arrays of the quickr backend", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(function(x, y) x + y)
    out <- f(nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
    expect_s3_class(out, "AnvlArray")
    expect_equal(backend(out), "quickr")
    expect_equal(as_array(out), array(c(5, 7, 9), dim = 3L))
    expect_equal(as_array(f(nv_scalar(1L), nv_scalar(2L))), 3L)
  })

  it("dispatches a repeated call to the compiled program", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(function(x, y) x + y)
    x <- nv_array(c(1, 2))
    y <- nv_array(c(10, 20))
    expect_equal(as_array(f(x, y)), array(c(11, 22), dim = 2L))
    f(x, y)
    expect_identical(jit_cache_size(f), 1L)
  })

  it("preserves a nested output structure", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(function(x) list(flags = x > 1L, payload = list(shifted = x + 1L)))
    out <- f(nv_array(1:3))
    expect_dtype(out$flags, "bool")
    expect_dtype(out$payload$shifted, "i32")
    expect_identical(as_array(out$flags), array(c(FALSE, TRUE, TRUE), dim = 3L))
    expect_identical(as_array(out$payload$shifted), array(2:4, dim = 3L))
  })

  it("returns plain R values with unwrap = TRUE", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(function(x) list(flags = x > 1L, payload = list(shifted = x + 1L)), unwrap = TRUE)
    out <- f(nv_array(1:3))
    expect_false(inherits(out$flags, "AnvlArray"))
    expect_false(inherits(out$payload$shifted, "AnvlArray"))
    expect_equal(as.logical(out$flags), c(FALSE, TRUE, TRUE))
    expect_equal(as.integer(out$payload$shifted), 2:4)
  })

  it("does not support donate", {
    skip_if_no_quickr()
    local_backend("quickr")
    expect_error(jit(function(x) x, donate = "x")(nv_array(1)), "donate", fixed = TRUE)
  })

  it("emits a call whose outputs are all empty directly", {
    skip_if_no_quickr()
    local_backend("quickr")
    # quickr's elementwise operators reject an empty operand even where both
    # shapes agree, so lowering `x + x` on a `2x0` failed. An empty result has
    # only one possible value, so it is emitted without touching the operands.
    empty <- nv_array(numeric(0), shape = c(2L, 0L))
    out <- jit(function(x) nv_add(x, x))(empty)
    expect_shape(out, c(2L, 0L))
    expect_equal(length(as.vector(out)), 0L)
  })

  it("computes NaN and infinite literals at runtime", {
    skip_if_no_quickr()
    local_backend("quickr")
    # quickr rejects NaN / +-Inf as literals in the code it compiles, so anvl
    # binds each one to a temp that computes it instead. Every route a special
    # float takes into a program goes through that.
    fill <- jit(function() nv_fill(NaN, shape = c(2L, 2L), dtype = "f64"))
    expect_true(all(is.nan(as.vector(fill()))))
    # ... a scalar fill takes the shapeless branch
    expect_equal(as.vector(jit(function() nv_fill(-Inf, shape = integer(), dtype = "f64"))()), -Inf)

    # As an operand of a call, where it is bound to a temp the call then reads.
    x <- nv_array(c(1, 2), dtype = "f64")
    expect_equal(as.vector(jit(function(v) nv_pmin(v, Inf))(x)), c(1, 2))
    expect_equal(as.vector(jit(function(v) nv_pmax(v, -Inf))(x)), c(1, 2))
    expect_true(all(is.nan(as.vector(jit(function(v) v + NaN)(x)))))

    # `nan_rm = TRUE` reductions seed themselves with +-Inf, the path that made
    # this necessary in the first place.
    y <- nv_array(c(1, NaN, 3), dtype = "f64")
    expect_equal(as.vector(nv_max(y, nan_rm = TRUE)), 3)
    expect_equal(as.vector(nv_min(y, nan_rm = TRUE)), 1)
    expect_equal(as.vector(nv_sum(y, nan_rm = TRUE)), 4)
  })

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
})

describe("graph_to_quickr_function()", {
  it("requires quickr", {
    graph <- trace_fn(function(x) x + x, list(x = nv_scalar(1.0, dtype = "f64")))
    if (!requireNamespace("quickr", quietly = TRUE)) {
      expect_error(graph_to_quickr_function(graph), "quickr")
      return()
    }
    skip_if_no_quickr()
    expect_equal(as_array(graph_to_quickr_function(graph)(2)), 4)
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
