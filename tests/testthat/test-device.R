test_that("nv_device builds a device of the backend in force", {
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
  expect_error(with_backend("plain", nv_device("cpu")), "plain")
})

test_that("nv_device rejects a device of another backend", {
  skip_if_no_quickr()
  skip_if(!pjrt::plugins_downloaded())
  dev <- with_backend("quickr", nv_device("cpu"))
  expect_error(nv_device(dev), "backend in force")
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
