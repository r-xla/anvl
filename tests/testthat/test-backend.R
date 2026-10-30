test_that("active_backend returns 'pjrt' by default", {
  be <- withr::with_options(list(anvl.backend = NULL), active_backend())
  expect_equal(be, "pjrt")
})

test_that("local_backend sets and restores the active backend", {
  old <- active_backend()
  local_backend("r")
  expect_equal(active_backend(), "r")
  expect_equal(backend(nv_array(1)), "r")
})

test_that("with_backend temporarily changes the backend", {
  expect_equal(active_backend(), "pjrt")
  result <- with_backend("r", {
    expect_equal(active_backend(), "r")
    backend(nv_array(1))
  })
  expect_equal(result, "r")
  expect_equal(active_backend(), "pjrt")
})

test_that("with_backend restores backend on error", {
  expect_equal(active_backend(), "pjrt")
  try(with_backend("r", stop("test error")), silent = TRUE)
  expect_equal(active_backend(), "pjrt")
})

describe("with_backend", {
  it("rejects the internal plain backend", {
    expect_error(with_backend("plain", 1), "plain")
    expect_error(local_backend("plain"), "plain")
    expect_equal(active_backend(), "pjrt")
  })
})

test_that("backend() returns the backend name", {
  expect_equal(backend(nv_array(1)), "pjrt")
})

test_that("backend() returns 'r' for r arrays", {
  local_backend("r")
  expect_equal(backend(nv_array(1)), "r")
})

test_that("nv_empty works with r backend", {
  local_backend("r")
  x <- nv_empty(c(0L, 3L), "f64")
  expect_equal(backend(x), "r")
  expect_dtype(x, "f64")
  expect_shape(x, c(0L, 3L))
})

test_that("nv_empty works with pjrt backend", {
  x <- nv_empty(c(0L, 3L), "f32")
  expect_equal(backend(x), "pjrt")
  expect_dtype(x, "f32")
  expect_shape(x, c(0L, 3L))
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

test_that("install_anvl installs nothing for the r backend", {
  expect_null(install_anvl("r"))
})

test_that("install_anvl rejects what is not a backend a user can select", {
  expect_error(install_anvl("plain"))
  expect_error(install_anvl("not-a-backend"))
})

describe("eager code", {
  it("rejects an array of another backend instead of guessing a default", {
    x <- with_backend("r", nv_array(1L))
    expect_error(x + 1.5, "\"r\"")
    expect_error(nv_fill_like(x, 0), "belongs to the .*\"r\".* backend")
  })
})

describe("the plain backend", {
  it("converts a bit64::integer64 payload by value", {
    out <- jit(function() nv_array(bit64::as.integer64(c(3, 5)), dtype = "i64"))()
    expect_equal(as.integer(as_array(out)), c(3L, 5L))
  })
})
