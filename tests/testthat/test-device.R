test_that("nv_device builds a device of the active backend", {
  skip_if_no_quickr()
  dev <- with_backend("quickr", nv_device("cpu"))
  expect_s3_class(dev, "QuickrDevice")
  expect_equal(backend(dev), "quickr")
  skip_if(!pjrt::plugins_downloaded())
  dev <- nv_device("cpu")
  expect_s3_class(dev, "PJRTDevice")
  expect_equal(backend(dev), "pjrt")
})

test_that("nv_device errors on the plain backend", {
  withr::local_options(anvl.backend = "plain")
  expect_error(nv_device("cpu"), "plain")
})

test_that("nv_device rejects a device of another backend", {
  skip_if_no_quickr()
  skip_if(!pjrt::plugins_downloaded())
  dev <- with_backend("quickr", nv_device("cpu"))
  expect_error(nv_device(dev), "active backend")
})

test_that("nv_device returns an existing device unchanged", {
  skip_if_no_quickr()
  with_backend("quickr", {
    dev <- nv_device("cpu")
    expect_identical(nv_device(dev), dev)
  })
})

test_that("default_device(backend = ...) uses the specified backend", {
  skip_if_no_quickr()
  expect_s3_class(default_device(backend = "quickr"), "QuickrDevice")
  skip_if(!pjrt::plugins_downloaded())
  expect_s3_class(default_device(backend = "pjrt"), "PJRTDevice")
})

test_that("is_device recognizes backend device objects", {
  expect_false(is_device("cpu"))
  expect_false(is_device(NULL))
  expect_false(is_device(1L))
  skip_if_no_quickr()
  expect_true(is_device(with_backend("quickr", nv_device("cpu"))))
  skip_if(!pjrt::plugins_downloaded())
  expect_true(is_device(nv_device("cpu")))
})

describe("default_device()", {
  it("returns the first CPU device when no option overrides it", {
    local_unset_default_device()
    expect_equal(default_device(), nv_device("cpu"))
  })

  it("falls back to the device ANVL_DEFAULT_DEVICE names", {
    local_unset_default_device()
    local_env_default("DEVICE", "cpu:1")
    expect_equal(default_device(), nv_device("cpu:1"))
  })

  it("prefers the `anvl.default_device` option over ANVL_DEFAULT_DEVICE", {
    local_unset_default_device()
    local_env_default("DEVICE", "cpu:1")
    with_default_device("cpu:0", expect_equal(default_device(), nv_device("cpu:0")))
  })

  it("returns the device the `anvl.default_device` option names", {
    with_default_device("cpu:1", expect_equal(default_device(), nv_device("cpu:1")))
  })

  it("resolves the option for the backend that asks", {
    skip_if_no_quickr()
    with_default_device("cpu", expect_equal(default_device(backend = "quickr"), quickr_device("cpu")))
  })
})

describe("local_default_device()", {
  it("places an array that names no device on that device", {
    dev <- nv_device("cpu:1")
    local_default_device(dev)
    expect_equal(device(nv_array(1:3)), dev)
  })

  it("compiles a jitted call that pins no device of its own for it", {
    local_default_device("cpu:1")
    expect_equal(device(jit(function() nv_fill(1, shape = 2L, dtype = "f64"))()), nv_device("cpu:1"))
  })

  it("resets the option at the end of the scope", {
    before <- getOption("anvl.default_device")
    local({
      local_default_device("cpu:1")
      expect_equal(getOption("anvl.default_device"), "cpu:1")
    })
    expect_equal(getOption("anvl.default_device"), before)
  })

  it("rejects something that is neither a device nor an identifier", {
    expect_error(local_default_device(1L), "device")
  })
})

describe("placement_device()", {
  it("is the device of a concrete array", {
    dev <- nv_device("cpu:1")
    expect_equal(placement_device(nv_array(1:3, device = dev)), dev)
  })

  it("is NULL for a constant of the trace, which names no device", {
    seen <- NULL
    jit(function(x) {
      seen <<- nv_array(c(1, 2), dtype = "f64")
      x
    })(nv_scalar(1, dtype = "f64"))
    expect_equal(backend(seen), "plain")
    expect_null(placement_device(seen))
  })

  it("is NULL for a value that is not an array at all", {
    expect_null(placement_device(1L))
    expect_null(placement_device(NULL))
  })
})
