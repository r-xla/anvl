describe("nv_scalar_like()", {
  it("takes the device of a concrete array", {
    dev <- nv_device("cpu:1")
    expect_equal(device(nv_scalar_like(nv_array(1:3, device = dev), 1L)), dev)
  })

  it("builds a constant of the trace from a constant of the trace", {
    # A constant of the trace is backend-agnostic: its `PlainDeviceCpu()` stands
    # in for a device rather than being one, so taking it for a placement would
    # allocate the scalar on the first CPU device.
    seen <- NULL
    f <- jit(function(x) {
      p <- nv_array(c(0.25, 0.75), dtype = "f64")
      seen <<- nv_scalar_like(p, 1)
      x + p
    })
    f(nv_scalar(1, dtype = "f64"))
    expect_equal(backend(seen), "plain")
  })
})

describe("the `_like` constructors", {
  # Regression: `nv_qnorm()` builds its coefficients with `nv_scalar_like()` from
  # an array built inside the trace. Reading that constant's `PlainDeviceCpu()`
  # back allocated them on `cpu:0` while the operands were elsewhere, and jit's
  # device autodetect then refused the graph -- which is what made every
  # `nv_qnorm()` call fail on CUDA while agreeing with base R on CPU.
  it("do not pin a trace to a device of their own", {
    dev <- nv_device("cpu:1")
    cases <- list(
      nv_array_like = function(p) nv_array_like(p, c(3, 4)),
      nv_scalar_like = function(p) nv_scalar_like(p, 3),
      nv_fill_like = function(p) nv_fill_like(p, 3),
      nv_iota_like = function(p) nv_iota_like(p, axis = 1L),
      # The values are uninitialized, so only the placement is asserted below.
      nv_empty_like = function(p) nv_empty_like(p)
    )
    for (nm in names(cases)) {
      f <- jit(function(x) x + cases[[nm]](nv_array(c(0.25, 0.75), dtype = "f64")))
      out <- f(nv_array(c(1, 2), dtype = "f64", device = dev))
      expect_equal(device(out), dev, info = nm)
    }
  })

  it("keep taking the shape and data type of a constant of the trace", {
    # Only the device is not a constant's to give: the rest still comes from it.
    seen <- NULL
    f <- jit(function(x) {
      seen <<- nv_fill_like(nv_array(c(1, 2, 3), dtype = "f32"), 1)
      x
    })
    f(nv_scalar(1, dtype = "f32"))
    expect_equal(shape(seen), 3L)
    expect_equal(dtype(seen), as_dtype("f32"))
  })
})
