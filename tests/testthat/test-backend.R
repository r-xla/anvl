test_that("default_backend returns 'pjrt' by default", {
  expect_equal(default_backend(), "pjrt")
})

test_that("local_backend sets and restores the default backend", {
  skip_if_no_quickr()
  old <- default_backend()
  local_backend("quickr")
  expect_equal(default_backend(), "quickr")
  expect_equal(backend(nv_array(1)), "quickr")
})

test_that("with_backend temporarily changes the backend", {
  skip_if_no_quickr()
  expect_equal(default_backend(), "pjrt")
  result <- with_backend("quickr", {
    expect_equal(default_backend(), "quickr")
    backend(nv_array(1))
  })
  expect_equal(result, "quickr")
  expect_equal(default_backend(), "pjrt")
})

test_that("with_backend restores backend on error", {
  skip_if_no_quickr()
  expect_equal(default_backend(), "pjrt")
  try(with_backend("quickr", stop("test error")), silent = TRUE)
  expect_equal(default_backend(), "pjrt")
})

test_that("backend() returns the backend name", {
  expect_equal(backend(nv_array(1)), "pjrt")
})

test_that("backend() returns 'quickr' for quickr arrays", {
  skip_if_no_quickr()
  local_backend("quickr")
  expect_equal(backend(nv_array(1)), "quickr")
})

test_that("nv_empty works with quickr backend", {
  skip_if_no_quickr()
  local_backend("quickr")
  x <- nv_empty("f64", c(0L, 3L))
  expect_equal(backend(x), "quickr")
  expect_equal(dtype(x), as_dtype("f64"))
  expect_equal(shape(x), c(0L, 3L))
})

test_that("nv_empty works with pjrt backend", {
  x <- nv_empty("f32", c(0L, 3L))
  expect_equal(backend(x), "pjrt")
  expect_equal(dtype(x), as_dtype("f32"))
  expect_equal(shape(x), c(0L, 3L))
})

test_that("install_anvl routes to the backend's installer and forwards ...", {
  args <- NULL
  local_mocked_bindings(
    install_pjrt = function(...) {
      args <<- list(...)
    },
    .package = "pjrt"
  )
  expect_null(install_anvl("pjrt", cuda = FALSE))
  expect_equal(args, list(cuda = FALSE))

  pkg <- NULL
  local_mocked_bindings(install.packages = function(pkgs, ...) {
    pkg <<- pkgs
  })
  expect_null(install_anvl("quickr"))
  expect_equal(pkg, "quickr")
})

test_that("install_anvl rejects backends that have nothing to install", {
  expect_error(install_anvl("plain"))
  expect_error(install_anvl("not-a-backend"))
})
