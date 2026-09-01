test_that("array", {
  x <- nv_array(1:4, dtype = "i32", shape = c(4, 1), device = "cpu")
  expect_snapshot(x)
  expect_class(x, "AnvlArray")
  expect_equal(shape(x), c(4, 1))
  expect_equal(dtype(x), as_dtype("i32"))
  expect_equal(as_array(x), array(1:4, c(4, 1)))
})

test_that("device returns the pjrt device", {
  x <- nv_array(1, device = "cpu")
  expect_true(device(x) == pjrt::as_pjrt_device("cpu"))
})

test_that("await on an AnvlArray returns the array invisibly", {
  x <- nv_array(1:4, dtype = "i32", device = "cpu")
  out <- withVisible(await(x))
  expect_false(out$visible)
  expect_identical(out$value, x)
})

test_that("nv_scalar", {
  x <- nv_scalar(1L, dtype = "f32", device = "cpu")
  x
  expect_class(x, "AnvlArray")
  expect_snapshot(x)
})

test_that("AbstractArray", {
  x <- AbstractArray(
    as_dtype("f32"),
    Shape(c(2, 3))
  )
  expect_snapshot(x)
  expect_true(inherits(x, "AbstractArray"))
  expect_true(eq_type(x, x))

  expect_false(
    eq_type(
      x,
      AbstractArray(
        as_dtype("f32"),
        Shape(c(2, 1))
      )
    )
  )

  expect_false(
    eq_type(
      x,
      AbstractArray(
        as_dtype("f64"),
        Shape(c(2, 3))
      )
    )
  )
})

test_that("ConcreteArray", {
  x <- ConcreteArray(
    nv_array(1:6, dtype = "f32", shape = c(2, 3), device = "cpu")
  )
  expect_true(inherits(x, "ConcreteArray"))
  expect_snapshot(x)
})

test_that("from DataType", {
  expect_class(nv_array(1L, "i32"), "AnvlArray")
  expect_class(nv_scalar(1L, "i32"), "AnvlArray")
  expect_class(nv_empty("i32", c(0, 1)), "AnvlArray")
})

test_that("nv_array from nv_array", {
  skip_if(!is_cuda())
  x <- nv_array(1, device = "cuda")
  expect_equal(platform(x), "cuda")
  expect_error(nv_array(x, device = "cpu"))
  expect_error(nv_array(x, shape = c(1, 1)))
  expect_error(nv_array(x, dtype = "f64"))
})

test_that("format", {
  expect_equal(format(nv_array(1:4, shape = c(4, 1))), "AnvlArray(dtype=i32, shape=4x1)")
})

test_that("nv_array(byrow = TRUE) fills row-major from a flat vector", {
  expect_equal(
    as_array(nv_array(1:6, shape = c(2L, 3L), byrow = TRUE)),
    matrix(1:6, nrow = 2, ncol = 3, byrow = TRUE)
  )
})

test_that("nv_array(byrow = TRUE) extends to higher-rank shapes", {
  # data values fill last axis fastest, mirroring row-major storage
  x <- nv_array(1:24, shape = c(2L, 3L, 4L), byrow = TRUE)
  expected <- aperm(array(1:24, dim = c(4L, 3L, 2L)), 3:1)
  expect_equal(as_array(x), expected)
  expect_equal(shape(x), c(2, 3, 4))
})

test_that("nv_array(byrow = TRUE) is a no-op for shapes with < 2 axes", {
  expect_equal(
    as_array(nv_array(1:4, byrow = TRUE)),
    as_array(nv_array(1:4))
  )
  expect_equal(
    as_array(nv_array(1L, shape = integer(), byrow = TRUE)),
    as_array(nv_array(1L, shape = integer()))
  )
})

test_that("nv_array(byrow = TRUE) re-fills a matrix input row-major", {
  # input is column-major matrix(1:6, 2, 3) but byrow re-interprets values
  expect_equal(
    as_array(nv_array(matrix(1:6, nrow = 2L), byrow = TRUE)),
    matrix(1:6, nrow = 2L, ncol = 3L, byrow = TRUE)
  )
})

test_that("nv_array(byrow = TRUE) errors when data is an AnvlArray", {
  x <- nv_array(1:6, shape = c(2L, 3L))
  expect_error(nv_array(x, byrow = TRUE), "byrow")
})

test_that("nv_matrix() infers ncol from nrow and data length", {
  expect_equal(
    as_array(nv_matrix(1:6, nrow = 2L)),
    matrix(1:6, nrow = 2L, ncol = 3L)
  )
})

test_that("nv_matrix() infers nrow from ncol and data length", {
  expect_equal(
    as_array(nv_matrix(1:6, ncol = 3L)),
    matrix(1:6, nrow = 2L, ncol = 3L)
  )
})

test_that("nv_matrix() accepts both nrow and ncol when consistent", {
  expect_equal(
    as_array(nv_matrix(1:6, nrow = 2L, ncol = 3L)),
    matrix(1:6, nrow = 2L, ncol = 3L)
  )
})

test_that("nv_matrix() forwards byrow to nv_array", {
  expect_equal(
    as_array(nv_matrix(1:6, nrow = 2L, byrow = TRUE)),
    matrix(1:6, nrow = 2L, ncol = 3L, byrow = TRUE)
  )
})

test_that("nv_matrix() forwards dtype to nv_array", {
  x <- nv_matrix(1:6, nrow = 2L, dtype = "f64")
  expect_equal(dtype(x), as_dtype("f64"))
  expect_equal(shape(x), c(2, 3))
})

test_that("nv_matrix() errors when neither nrow nor ncol is supplied", {
  expect_error(nv_matrix(1:6), "nrow.*ncol")
})

test_that("nv_matrix() errors when data length is not divisible by nrow/ncol", {
  expect_error(nv_matrix(1:7, nrow = 2L), "not a multiple")
  expect_error(nv_matrix(1:7, ncol = 2L), "not a multiple")
})

test_that("nv_matrix() errors when nrow * ncol does not match data length", {
  expect_error(nv_matrix(1:6, nrow = 2L, ncol = 4L), "does not match")
})

test_that("nv_matrix() handles existing AnvlArray inputs", {
  x <- nv_array(1:6, shape = c(2L, 3L))
  expect_equal(as_array(nv_matrix(x, nrow = 2L)), as_array(x))
  expect_error(nv_matrix(x, nrow = 3L), "Cannot change shape")
})

test_that("nv_matrix() handles zero-row / zero-column shapes", {
  x0 <- nv_matrix(integer(0), nrow = 0L)
  expect_equal(shape(x0), c(0, 0))
  x1 <- nv_matrix(integer(0), ncol = 3L)
  expect_equal(shape(x1), c(0, 3))
})

test_that("nv_matrix() recycles scalar data like base matrix()", {
  expect_equal(
    as_array(nv_matrix(1, nrow = 3L, ncol = 3L)),
    matrix(1, nrow = 3L, ncol = 3L)
  )
  expect_equal(
    as_array(nv_matrix(1L, nrow = 2L, ncol = 4L)),
    matrix(1L, nrow = 2L, ncol = 4L)
  )
  expect_equal(
    as_array(nv_matrix(TRUE, nrow = 2L, ncol = 2L)),
    matrix(TRUE, nrow = 2L, ncol = 2L)
  )
})

test_that("nv_matrix() with scalar data defaults missing axis to 1", {
  expect_equal(
    as_array(nv_matrix(1, nrow = 3L)),
    matrix(1, nrow = 3L, ncol = 1L)
  )
  expect_equal(
    as_array(nv_matrix(1, ncol = 3L)),
    matrix(1, nrow = 1L, ncol = 3L)
  )
  expect_equal(
    as_array(nv_matrix(1)),
    matrix(1)
  )
})

test_that("nv_matrix() broadcasts a scalar AnvlArray", {
  expect_equal(
    as_array(nv_matrix(nv_scalar(1), nrow = 3L, ncol = 3L)),
    matrix(1, nrow = 3L, ncol = 3L)
  )
})
test_that("== and != operators throw errors for AbstractArray", {
  x <- AbstractArray("f32", 1L)
  y <- AbstractArray("f32", 1L)
  expect_error(x == y, "Use.*eq_type")
  expect_error(x != y, "Use.*neq_type")
})

test_that("to_abstract", {
  # an R value, which has no dtype of its own yet
  expect_equal(to_abstract(TRUE), RData(integer(), "logical"))
  expect_equal(to_abstract(1L), RData(integer(), "integer"))
  expect_equal(to_abstract(1.0), RData(integer(), "double"))
  expect_equal(to_abstract(array(1:6, c(2, 3))), RData(c(2L, 3L), "integer"))
  # anvl array
  x <- nv_array(1:4, dtype = "f32", shape = c(2, 2))
  expect_equal(to_abstract(x), ConcreteArray(x))
  # graph box
  aval <- GraphValue(AbstractArray("f32", c(2, 2)))
  x <- GraphBox(aval, local_descriptor())
  expect_equal(to_abstract(x), aval$aval)

  # pure -- an R value contributes the dtype it would commit to
  x <- nv_scalar(1)
  expect_equal(to_abstract(x, pure = TRUE), AbstractArray("f32", c()))
  expect_equal(to_abstract(1L, pure = TRUE), AbstractArray("i32", c()))
  expect_error(to_abstract(list(1, 2)), "is not an array-like object")
})


test_that("as_shape for c() (i.e., NULL)", {
  expect_equal(as_shape(c()), Shape(integer()))
})

test_that("nv_aval creates AbstractArray", {
  expect_equal(
    nv_aval("f32", c()),
    AbstractArray("f32", Shape(integer()))
  )
  expect_equal(
    nv_aval(as_dtype("i32"), 1:2),
    AbstractArray("i32", Shape(1:2))
  )
})

test_that("nv_aval creates RData from an R storage type", {
  expect_equal(nv_aval("double", c()), RData(integer(), "double"))
  expect_equal(nv_aval("integer", 1:2), RData(1:2, "integer"))
  expect_equal(nv_aval("logical", c()), RData(integer(), "logical"))
  # the value is deliberately unknown, and there is no dtype to report
  expect_error(dtype(nv_aval("double", c())), "no data type of its own")
  expect_equal(shape(nv_aval("double", 1:2)), 1:2)
  # it traces like any other input aval
  graph <- trace_fn(function(x) x + nv_scalar(1, dtype = "f64"), list(x = nv_aval("double", integer())))
  expect_s3_class(graph$inputs[[1L]]$aval, "AbstractArray")
  expect_equal(dtype(graph$inputs[[1L]]$aval), as_dtype("f64"))
  expect_equal(graph$rdata_types, "double")
})

test_that("as_shape for c() (i.e., NULL)", {
  expect_equal(as_shape(c()), Shape(integer()))
})


test_that("stablehlo dtype is printed", {
  skip_if(!is_cpu())
  expect_snapshot(nv_array(TRUE))
})

test_that("quickr_device is a classed object", {
  skip_if_no_quickr()
  dev <- quickr_device("cpu")
  expect_s3_class(dev, "QuickrDevice")
  expect_equal(format(dev), "QuickrDevice(cpu)")
  expect_equal(as.character(dev), "cpu")
})

test_that("PlainDeviceCpu is a classed object", {
  dev <- PlainDeviceCpu()
  expect_s3_class(dev, "PlainDeviceCpu")
  expect_equal(format(dev), "PlainDeviceCpu")
  expect_equal(as.character(dev), "cpu")
})

test_that("device returns QuickrDevice for quickr arrays", {
  skip_if_no_quickr()
  local_backend("quickr")
  x <- nv_array(1)
  dev <- device(x)
  expect_s3_class(dev, "QuickrDevice")
})

test_that("device returns PlainDeviceCpu for plain arrays", {
  x <- globals$backends[["plain"]]$new_data(1, "f32", 1L, NULL)
  dev <- device(x)
  expect_s3_class(dev, "PlainDeviceCpu")
})

test_that("platform returns 'cpu' for quickr backend", {
  skip_if_no_quickr()
  local_backend("quickr")
  expect_equal(platform(nv_array(1)), "cpu")
})

test_that("platform returns 'cpu' for plain backend", {
  x <- globals$backends[["plain"]]$new_data(1, "f32", 1L, NULL)
  expect_equal(platform(x), "cpu")
})

test_that("nv_array respects backend argument", {
  skip_if_no_quickr()
  local_backend("quickr")
  x <- nv_array(1, backend = "pjrt")
  expect_equal(backend(x), "pjrt")
})

test_that("nv_array infers backend from device object", {
  skip_if_no_quickr()
  local_backend("quickr")
  x <- nv_array(1, device = pjrt::pjrt_device("cpu"))
  expect_equal(backend(x), "pjrt")
  expect_equal(device(x), nv_device("cpu", "pjrt"))
})

test_that("nv_array errors when backend specified inside jit", {
  expect_error(
    jit(function() nv_array(1, backend = "pjrt"))(),
    "must not be specified"
  )
})

test_that("default floating dtype is f32 for pjrt", {
  expect_equal(dtype(nv_array(1.0)), as_dtype("f32"))
  expect_equal(dtype(nv_scalar(1.0)), as_dtype("f32"))
})

test_that("default floating dtype is f64 for quickr", {
  skip_if_no_quickr()
  local_backend("quickr")
  expect_equal(dtype(nv_array(1.0)), as_dtype("f64"))
  expect_equal(dtype(nv_scalar(1.0)), as_dtype("f64"))
})

test_that("nv_array_like inherits dtype, shape, device, backend from like", {
  like <- nv_array(c(1L, 2L, 3L), dtype = "i16")
  out <- nv_array_like(like, c(7L, 8L, 9L))
  expect_equal(dtype(out), dtype(like))
  expect_equal(shape(out), shape(like))
  expect_equal(backend(out), backend(like))
  expect_equal(as.integer(out), c(7L, 8L, 9L))
})

test_that("nv_array_like respects explicit overrides", {
  like <- nv_array(c(1L, 2L, 3L), dtype = "i16")
  out <- nv_array_like(like, c(1L, 2L, 3L, 4L), dtype = "i32", shape = 4L)
  expect_equal(dtype(out), as_dtype("i32"))
  expect_equal(shape(out), 4L)
})

test_that("nv_scalar_like inherits dtype, device, backend from like", {
  like <- nv_scalar(1L, dtype = "i16")
  out <- nv_scalar_like(like, 7L)
  expect_equal(dtype(out), dtype(like))
  expect_equal(shape(out), integer())
  expect_equal(backend(out), backend(like))
  expect_equal(as.integer(out), 7L)
})

describe("as_anvl_array", {
  it("passes AnvlArrays through unchanged", {
    x <- nv_array(1:3)
    expect_identical(as_anvl_array(x), x)
  })

  it("converts scalar R literals into scalar AnvlArrays", {
    out <- as_anvl_array(1L)
    expect_s3_class(out, "AnvlArray")
    expect_equal(shape(out), integer())
    expect_equal(dtype(out), as_dtype("i32"))
    expect_equal(dtype(as_anvl_array(1)), as_dtype("f32"))
    expect_equal(dtype(as_anvl_array(TRUE)), as_dtype("bool"))
  })

  it("converts R arrays into AnvlArrays preserving shape", {
    out <- as_anvl_array(array(1:6, c(2, 3)))
    expect_s3_class(out, "AnvlArray")
    expect_equal(shape(out), c(2L, 3L))
    expect_equal(dtype(out), as_dtype("i32"))
  })

  it("places R literals on the requested device", {
    dev <- nv_device("cpu:1", "pjrt")
    expect_equal(device(as_anvl_array(1L, device = dev)), dev)
  })

  it("converts under jit() too, at the same dtype", {
    # The name says it converts; it must not mean one thing here and another
    # inside a trace.
    seen <- list()
    f <- jit(function(x) {
      out <- as_anvl_array(x)
      seen[["dtype"]] <<- dtype(out)
      seen[["shape"]] <<- shape(out)
      out + 1
    })
    invisible(f(1))
    expect_equal(seen$dtype, dtype(as_anvl_array(1)))
    expect_equal(seen$shape, shape(as_anvl_array(1)))
    # ... including a value that was already a traced R value.
    expect_equal(dtype(jit(function(x) as_anvl_array(x))(1L)), as_dtype("i32"))
  })

  it("errors if an AnvlArray is on a different device than requested", {
    dev0 <- nv_device("cpu:0", "pjrt")
    dev1 <- nv_device("cpu:1", "pjrt")
    x <- nv_array(1:3, device = dev0)
    expect_error(
      as_anvl_array(x, device = dev1),
      "unexpected device"
    )
  })

  it("rejects non-arrayish inputs", {
    expect_error(as_anvl_array("foo"), "Expected arrayish")
    expect_error(as_anvl_array(list()), "Expected arrayish")
  })

  it("passes traced boxes through unchanged under jit()", {
    f <- jit(function(x) {
      y <- as_anvl_array(x)
      y + 1
    })
    out <- f(nv_array(1:3))
    expect_equal(as_array(out), array(2:4, dim = 3L))
  })

  it("handles R literals under jit()", {
    f <- jit(function() as_anvl_array(1L) + 1L)
    out <- f()
    expect_s3_class(out, "AnvlArray")
    expect_equal(as.integer(out), 2L)
  })

  it("converting at the default is what a promote rule avoids", {
    # `as_anvl_array()` has nothing to decide the dtype from, so an R value takes
    # its default and the digits below it are gone -- identically in both modes,
    # which is the contract. `as_anvl_arrays(.promote = )` decides and converts in
    # one step, and is exact.
    x <- nv_scalar(1, dtype = "f64")
    at_default <- as.numeric(as_array(x * as_anvl_array(sqrt(2))))
    expect_false(identical(at_default, sqrt(2)))
    expect_identical(as_array(jit(function(t) x * as_anvl_array(t))(sqrt(2))), at_default)

    promoted <- as_anvl_arrays(x = x, y = sqrt(2), .promote = promote_like("x"))
    expect_identical(as_array(promoted$y), sqrt(2))
  })
})

describe("as_anvl_arrays", {
  it("places R literals on the first concrete input's device", {
    dev <- nv_device("cpu:1", "pjrt")
    x <- nv_array(1:3, device = dev)
    out <- as_anvl_arrays(x, 1L)
    expect_equal(device(out[[1L]]), dev)
    expect_equal(device(out[[2L]]), dev)
    # ... and when a promote rule realizes them
    out <- as_anvl_arrays(x, 1.5, .promote = promote_common())
    expect_equal(device(out[[2L]]), dev)
  })

  it("uses the default device when no concrete input is present", {
    out <- as_anvl_arrays(1L, 2L)
    expect_equal(device(out[[1L]]), default_device())
    expect_equal(device(out[[2L]]), default_device())
  })

  it("converts R values at their default when no rule decides one", {
    out <- as_anvl_arrays(nv_array(1:3), 1L, 1.5, TRUE)
    expect_s3_class(out[[2L]], "AnvlArray")
    expect_equal(lapply(out[-1L], dtype), lapply(c("i32", "f32", "bool"), as_dtype))
  })

  it("keeps the names of the arguments", {
    out <- as_anvl_arrays(x = nv_array(1:3), y = 1L)
    expect_named(out, c("x", "y"))
  })

  it("errors when concrete inputs live on different devices", {
    dev0 <- nv_device("cpu:0", "pjrt")
    dev1 <- nv_device("cpu:1", "pjrt")
    x <- nv_array(1:3, device = dev0)
    y <- nv_array(1:3, device = dev1)
    expect_error(
      as_anvl_arrays(x, y),
      "multiple devices"
    )
  })

  it("errors when concrete inputs come from different backends", {
    skip_if_no_quickr()
    dev_pjrt <- nv_device("cpu", "pjrt")
    dev_quickr <- nv_device("cpu", "quickr")
    x <- nv_array(1:3, device = dev_pjrt)
    y <- nv_array(1:3, device = dev_quickr)
    expect_error(
      as_anvl_arrays(x, y),
      "multiple backends"
    )
  })

  it("passes concrete inputs on the same device through unchanged", {
    x <- nv_array(1:3)
    y <- nv_array(4:6)
    out <- as_anvl_arrays(x, y)
    expect_identical(out[[1L]], x)
    expect_identical(out[[2L]], y)
  })

  it("canonicalizes mixed traced and literal inputs under jit()", {
    f <- jit(function(x) {
      args <- as_anvl_arrays(x, 1L)
      args[[1L]] + args[[2L]]
    })
    out <- f(nv_array(1:3))
    expect_equal(as.integer(out), 2:4)
  })

  it("canonicalizes multiple traced inputs under jit()", {
    f <- jit(function(x, y) {
      args <- as_anvl_arrays(x, y)
      args[[1L]] + args[[2L]]
    })
    out <- f(nv_array(1:3), nv_array(4:6))
    expect_equal(as.integer(out), c(5L, 7L, 9L))
  })

  it("leaves dtypes alone unless asked to promote", {
    out <- as_anvl_arrays(nv_array(1L), nv_array(1.5))
    expect_identical(as.character(dtype(out[[1L]])), "i32")
    expect_identical(as.character(dtype(out[[2L]])), "f32")
  })

  # REVIEW: THe promotion tests should be in test-promotion.R
  # One block for each promotion rule (promote_common, promote_yield, etc.)

  it("realizes every input at the common dtype with promote_common()", {
    out <- as_anvl_arrays(nv_array(1L), nv_array(1.5), .promote = promote_common())
    expect_identical(as.character(dtype(out[[1L]])), "f32")
    expect_identical(as.character(dtype(out[[2L]])), "f32")
    expect_equal(as.numeric(out[[1L]]), 1)
    expect_equal(as.numeric(out[[2L]]), 1.5)
  })

  it("settles R values at the promoted dtype, not at their default", {
    # A bare R value carries no dtype until something decides one. Promoting is
    # that decision, and it is the one that reaches the value.
    out <- as_anvl_arrays(nv_array(c(1, 2), dtype = "f64"), 2L, .promote = promote_common())
    expect_identical(as.character(dtype(out[[1L]])), "f64")
    expect_identical(as.character(dtype(out[[2L]])), "f64")
    expect_equal(as.numeric(out[[2L]]), 2)
  })

  it("builds an R value at the common dtype rather than converting to it", {
    # The point of realize_at(): converting an f32 sqrt(2) to f64 would only
    # widen a number that had already lost its digits.
    out <- as_anvl_arrays(nv_array(1, dtype = "f64"), sqrt(2), .promote = promote_common())
    expect_identical(as.character(dtype(out[[2L]])), "f64")
    expect_equal(as.numeric(out[[2L]]), sqrt(2), tolerance = 1e-15)
  })

  it("promotes R literals onto the aligned device", {
    dev <- nv_device("cpu:1", "pjrt")
    x <- nv_array(c(1, 2), dtype = "f32", device = dev)
    out <- as_anvl_arrays(x, 2L, .promote = promote_common())
    expect_equal(device(out[[1L]]), dev)
    expect_equal(device(out[[2L]]), dev)
  })

  it("realizes every input at one argument's dtype with a named anchor", {
    out <- as_anvl_arrays(x = nv_array(1L), y = nv_array(2L, dtype = "i8"), .promote = promote_like("x"))
    expect_identical(as.character(dtype(out$x)), "i32")
    # REVIEW(TODO): Add issue to add helper, so expect_dtype(out$x, "i32") works.
    expect_identical(as.character(dtype(out$y)), "i32")
  })

  it("refuses an input the anchor's dtype cannot hold, unless forced", {
    expect_error(
      as_anvl_arrays(x = nv_array(1L), y = nv_array(1.5), .promote = promote_like("x")),
      "not promotable"
    )
    expect_error(
      as_anvl_arrays(x = nv_array(1L), y = 1.5, .promote = promote_like("x")),
      "R double"
    )
    # ... which is a conversion, not a promotion: `force` narrows where it has to
    out <- as_anvl_arrays(
      x = nv_array(1L),
      y = nv_array(1.5),
      .promote = promote_like("x", force = TRUE)
    )
    expect_identical(as.character(dtype(out$y)), "i32")
    expect_equal(as.integer(out$y), 1L)
    # an R value, forced, is still built where it is exact and converted by the
    # program rather than by R
    out <- as_anvl_arrays(x = nv_array(1L), y = 1.9, .promote = promote_like("x", force = TRUE))
    expect_equal(as.integer(out$y), 1L)
  })

  it("accepts the anchor by position too", {
    out <- as_anvl_arrays(nv_array(1L), nv_array(1.5), .promote = promote_like(2))
    expect_identical(as.character(dtype(out[[1L]])), "f32")
    expect_identical(as.character(dtype(out[[2L]])), "f32")
    expect_identical(
      unname(lapply(as_anvl_arrays(x = nv_array(1L), y = nv_array(1.5), .promote = promote_like("y")), dtype)),
      lapply(as_anvl_arrays(nv_array(1L), nv_array(1.5), .promote = promote_like(2L)), dtype)
    )
  })

  it("builds an R value at the anchor's dtype rather than converting to it", {
    out <- as_anvl_arrays(x = nv_array(1, dtype = "f64"), y = sqrt(2), .promote = promote_like("x"))
    expect_identical(as.character(dtype(out$y)), "f64")
    expect_identical(as.numeric(out$y), sqrt(2))
  })

  it("commits an R value anchor to its default dtype", {
    # REVIEW: Needs to be properly documented. Answer here: Do we rely on this somewhere?
    out <- as_anvl_arrays(x = 1L, y = nv_array(2L), .promote = promote_like("x"))
    expect_identical(as.character(dtype(out$x)), "i32")
    expect_identical(as.character(dtype(out$y)), "i32")
  })

  it("rejects an argument reference that names nothing", {
    expect_error(as_anvl_arrays(x = nv_array(1L), .promote = promote_like("y")), "does not have")
    expect_error(as_anvl_arrays(nv_array(1L), .promote = promote_like("x")), "does not have")
    expect_error(as_anvl_arrays(nv_array(1L), .promote = promote_like(2)), "not <= 1")
    expect_error(
      as_anvl_arrays(x = nv_array(1L), .promote = promote_common(only = c("x", "z"))),
      "does not have"
    )
    expect_error(promote_like(TRUE), "name or position")
    expect_error(promote_common(only = list()), "names or positions")
  })

  it("realizes only the inputs a rule names, aligning the rest", {
    # `nv_ifelse()`'s shape: `pred` takes part in the device alignment but not in
    # the promotion, so it stays a bool.
    out <- as_anvl_arrays(
      pred = nv_array(TRUE),
      a = nv_array(1L, dtype = "i8"),
      b = 3L,
      # REVIEW: `only` is an odd name.
      .promote = promote_common(only = c("a", "b"))
    )
    # `pred` keeps out of it: had it taken part, the common dtype would have
    # reached it and prim_ifelse() would have been handed a non-bool predicate.
    expect_identical(as.character(dtype(out$pred)), "bool")
    expect_identical(as.character(dtype(out$a)), "i8")
    expect_identical(as.character(dtype(out$b)), "i8")
    # ... and `only` takes positions too
    by_position <- as_anvl_arrays(
      nv_array(TRUE),
      nv_array(1L, dtype = "i8"),
      3L,
      .promote = promote_common(only = 2:3)
    )
    expect_identical(unname(lapply(out, dtype)), lapply(by_position, dtype))
    # An excluded R value is still converted, and still lands on the shared device
    dev <- nv_device("cpu:1", "pjrt")
    out <- as_anvl_arrays(
      pred = TRUE,
      a = nv_array(1L, device = dev),
      b = 1.5,
      .promote = promote_common(only = c("a", "b"))
    )
    expect_s3_class(out$pred, "AnvlArray")
    expect_equal(device(out$pred), dev)
  })

  it("realizes every input at a dtype the caller names", {
    out <- as_anvl_arrays(nv_array(1L), sqrt(2), .promote = promote_dtype("f64"))
    expect_identical(as.character(dtype(out[[1L]])), "f64")
    expect_identical(as.character(dtype(out[[2L]])), "f64")
    # built at f64, not converted from an f32 of it
    expect_identical(as.numeric(out[[2L]]), sqrt(2))
    # ... and it says what the result type is rather than negotiating it --
    # but refuses an input that data type cannot hold unless told to force it
    expect_error(
      as_anvl_arrays(nv_array(1, dtype = "f64"), .promote = promote_dtype("i32")),
      "not promotable"
    )
    out <- as_anvl_arrays(nv_array(1, dtype = "f64"), .promote = promote_dtype("i32", force = TRUE))
    expect_identical(as.character(dtype(out[[1L]])), "i32")
  })

  it("prints what a rule is", {
    expect_equal(format(promote_common()), "<promote_common>")
    expect_equal(format(promote_like("x")), "<promote_like(\"x\")>")
    expect_equal(format(promote_dtype("f64")), "<promote_dtype(f64)>")
    expect_equal(format(promote_like("x", force = TRUE)), "<promote_like(\"x\", force)>")
    expect_equal(format(promote_dtype("f64", force = TRUE)), "<promote_dtype(f64, force)>")
    expect_equal(format(promote_common(only = c("a", "b"))), "<promote_common only \"a\", \"b\">")
    expect_output(print(promote_common()), "promote_common")
  })

  it("anchors under jit() as well", {
    f <- jit(function(x, y) {
      args <- as_anvl_arrays(x = x, y = y, .promote = promote_like("x"))
      args$x + args$y
    })
    out <- f(nv_array(1L), nv_array(2L, dtype = "i8"))
    expect_equal(dtype(out), as_dtype("i32"))
  })

  it("agrees with nv_promote_to_common(), which shares its implementation", {
    x <- nv_array(1L)
    y <- nv_array(1.5)
    expect_equal(
      lapply(as_anvl_arrays(x, y, .promote = promote_common()), as.numeric),
      lapply(nv_promote_to_common(x, y), as.numeric)
    )
  })

  it("promotes under jit() as well", {
    f <- jit(function(x, y) {
      args <- as_anvl_arrays(x, y, .promote = promote_common())
      args[[1L]] + args[[2L]]
    })
    out <- f(nv_array(1L), nv_array(1.5))
    expect_identical(as.character(dtype(out)), "f32")
    expect_equal(as.numeric(out), 2.5)
  })

  it("promotes several groups independently with promote_grouped()", {
    out <- as_anvl_arrays(
      x = nv_array(1L),
      y = 1.5,
      a = nv_array(1L, dtype = "i8"),
      b = 2L,
      .promote = promote_grouped(
        promote_common(only = c("x", "y")),
        promote_common(only = c("a", "b"))
      )
    )
    expect_identical(as.character(dtype(out$x)), "f32")
    expect_identical(as.character(dtype(out$y)), "f32")
    expect_identical(as.character(dtype(out$a)), "i8")
    expect_identical(as.character(dtype(out$b)), "i8")

    # The groups may use different kinds of rule, and may be named by position.
    out <- as_anvl_arrays(
      nv_array(1L),
      1.5,
      nv_array(1L, dtype = "i8"),
      2L,
      .promote = promote_grouped(promote_dtype("f64", only = 1:2), promote_like(3, only = 3:4))
    )
    expect_identical(lapply(out, function(z) as.character(dtype(z))), list("f64", "f64", "i8", "i8"))

    # An argument no rule names is aligned and converted, as without a rule.
    out <- as_anvl_arrays(
      x = nv_array(1L, dtype = "i8"),
      y = 3L,
      z = 1.5,
      .promote = promote_grouped(promote_common(only = c("x", "y")))
    )
    expect_identical(as.character(dtype(out$y)), "i8")
    expect_identical(as.character(dtype(out$z)), "f32")

    # A group of one is the plain case.
    expect_identical(
      lapply(as_anvl_arrays(nv_array(1L), 1.5, .promote = promote_grouped(promote_common())), dtype),
      lapply(as_anvl_arrays(nv_array(1L), 1.5, .promote = promote_common()), dtype)
    )
  })

  it("resolves every rule before applying any of them", {
    # `b` takes `a`'s dtype, and `a` is itself promoted -- the second rule must
    # still see the dtype `a` had when the call was made.
    out <- as_anvl_arrays(
      a = nv_array(1L, dtype = "i8"),
      b = 2L,
      c = nv_array(1, dtype = "f64"),
      .promote = promote_grouped(
        promote_like("a", only = c("a", "b")),
        promote_dtype("f32", only = "c", force = TRUE)
      )
    )
    expect_identical(as.character(dtype(out$b)), "i8")
    expect_identical(as.character(dtype(out$c)), "f32")
  })

  it("rejects rules that cover the same argument twice", {
    expect_error(
      as_anvl_arrays(
        x = nv_array(1L),
        y = 1.5,
        .promote = promote_grouped(promote_common(only = "x"), promote_common(only = c("x", "y")))
      ),
      "covers the same argument"
    )
    # A rule with no `only` covers everything, so it cannot share a call.
    expect_error(
      as_anvl_arrays(
        x = nv_array(1L),
        y = 1.5,
        .promote = promote_grouped(promote_common(only = "x"), promote_common())
      ),
      "covers the same argument"
    )
  })

  it("does not capture an input that is named like its own argument", {
    # The inputs go in `...`, so the rule's argument is dotted: an input a caller
    # happens to call `promote` has to stay data.
    out <- as_anvl_arrays(promote = 1L, x = nv_array(1.5))
    expect_named(out, c("promote", "x"))
    expect_s3_class(out$promote, "AnvlArray")
    expect_identical(as.character(dtype(out$promote)), "i32")
    # ... and it can even be the one a rule points at
    out <- as_anvl_arrays(promote = nv_array(1L), x = 2L, .promote = promote_like("promote"))
    expect_identical(as.character(dtype(out$x)), "i32")
  })

  it("rejects anything that is not a promotion rule", {
    expect_error(as_anvl_arrays(nv_array(1L), .promote = TRUE), "must be a promotion rule")
  })
})

describe("arr", {
  it("creates 1D vector when no shape is specified", {
    expect_equal(
      arr(1, 2, 3),
      array(c(1, 2, 3))
    )
    expect_equal(arr(1), array(1))
  })

  it("creates an array with the requested shape", {
    expect_equal(
      arr(1, 2, 3, 4, shape = c(2, 2)),
      array(1:4, dim = c(2, 2))
    )
  })

  it("recycles a single value to fill the requested shape", {
    expect_equal(
      arr(1, shape = c(2, 2)),
      array(1, dim = c(2, 2))
    )
  })

  it("errors when number of values does not match shape", {
    expect_error(
      arr(1, 2, shape = c(2, 2)),
      "Number of elements is 2"
    )
  })

  it("errors when no values are supplied", {
    expect_error(arr(), "Invalid input values")
  })

  it("errors when shape is not integerish", {
    expect_error(arr(1, 2, shape = "foo"))
  })
})
