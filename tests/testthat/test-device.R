test_that("nv_device dispatches to the backend-specific constructor", {
  skip_if_no_quickr()
  dev <- nv_device("cpu", backend = "quickr")
  expect_s3_class(dev, "QuickrDevice")
  expect_equal(backend(dev), "quickr")
})

test_that("nv_device returns PJRTDevice for pjrt backend", {
  skip_if(!pjrt::plugins_downloaded())
  dev <- nv_device("cpu", backend = "pjrt")
  expect_s3_class(dev, "PJRTDevice")
  expect_equal(backend(dev), "pjrt")
})

test_that("nv_device errors on the plain backend", {
  expect_error(nv_device("cpu", backend = "plain"), "plain")
})

test_that("nv_device errors when pass-through device has mismatched backend", {
  skip_if_no_quickr()
  skip_if(!pjrt::plugins_downloaded())
  dev <- nv_device("cpu", backend = "quickr")
  expect_error(
    nv_device(dev, backend = "pjrt"),
    "has backend"
  )
})

test_that("nv_device returns an existing device unchanged", {
  skip_if_no_quickr()
  dev <- nv_device("cpu", backend = "quickr")
  expect_identical(nv_device(dev), dev)
  expect_identical(nv_device(dev, backend = "quickr"), dev)
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
  expect_true(is_device(nv_device("cpu", backend = "quickr")))
  skip_if(!pjrt::plugins_downloaded())
  expect_true(is_device(nv_device("cpu", backend = "pjrt")))
})
