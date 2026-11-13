test_that("active_backend returns 'pjrt' by default", {
  be <- withr::with_options(list(anvl.backend = NULL), active_backend())
  expect_equal(be, "pjrt")
})

test_that("local_backend sets and restores the active backend", {
  skip_if_no_quickr()
  old <- active_backend()
  local_backend("quickr")
  expect_equal(active_backend(), "quickr")
  expect_equal(backend(nv_array(1)), "quickr")
})

test_that("with_backend temporarily changes the backend", {
  skip_if_no_quickr()
  expect_equal(active_backend(), "pjrt")
  result <- with_backend("quickr", {
    expect_equal(active_backend(), "quickr")
    backend(nv_array(1))
  })
  expect_equal(result, "quickr")
  expect_equal(active_backend(), "pjrt")
})

test_that("with_backend restores backend on error", {
  skip_if_no_quickr()
  expect_equal(active_backend(), "pjrt")
  try(with_backend("quickr", stop("test error")), silent = TRUE)
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

test_that("backend() returns 'quickr' for quickr arrays", {
  skip_if_no_quickr()
  local_backend("quickr")
  expect_equal(backend(nv_array(1)), "quickr")
})

test_that("nv_empty works with quickr backend", {
  skip_if_no_quickr()
  local_backend("quickr")
  x <- nv_empty(c(0L, 3L), "f64")
  expect_equal(backend(x), "quickr")
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

test_that("install_anvl rejects backends that have nothing to install", {
  expect_error(install_anvl("plain"))
  expect_error(install_anvl("not-a-backend"))
})

describe("skip_if_no_pjrt_plugin", {
  # An empty plugin cache and none of the variables that make a plugin
  # available otherwise. The plugin this session already loaded is unaffected.
  local_no_pjrt_plugin <- function(envir = parent.frame()) {
    withr::local_envvar(
      R_USER_CACHE_DIR = withr::local_tempdir(.local_envir = envir),
      PJRT_PLUGIN_PATH_CPU = NA,
      PJRT_PLUGIN_PATH_CUDA = NA,
      PJRT_INSTALL = NA,
      .local_envir = envir
    )
  }

  it("skips when the plugin is neither downloaded, pointed to, nor allowed to download", {
    local_no_pjrt_plugin()
    expect_condition(skip_if_no_pjrt_plugin("cpu"), class = "skip")
    withr::local_envvar(PJRT_INSTALL = "0")
    expect_condition(skip_if_no_pjrt_plugin("cpu"), class = "skip")
  })

  it("does not skip when the plugin is downloaded", {
    local_no_pjrt_plugin()
    dir.create(file.path(tools::R_user_dir("pjrt", "cache"), "cpu"), recursive = TRUE)
    expect_no_condition(skip_if_no_pjrt_plugin("cpu"), class = "skip")
  })

  it("does not skip when PJRT_PLUGIN_PATH_<PLATFORM> points to the plugin", {
    local_no_pjrt_plugin()
    withr::local_envvar(PJRT_PLUGIN_PATH_CPU = "libpjrt_cpu.so")
    expect_no_condition(skip_if_no_pjrt_plugin("cpu"), class = "skip")
    expect_condition(skip_if_no_pjrt_plugin("cuda"), class = "skip")
  })

  it("does not skip when PJRT_INSTALL=1 allows downloading the plugin", {
    local_no_pjrt_plugin()
    withr::local_envvar(PJRT_INSTALL = "1")
    expect_no_condition(skip_if_no_pjrt_plugin("cpu"), class = "skip")
  })

  it("defaults to the platform of the default device", {
    local_no_pjrt_plugin()
    withr::local_envvar(PJRT_PLUGIN_PATH_CPU = "libpjrt_cpu.so")
    local_default_device("cuda:1")
    expect_condition(skip_if_no_pjrt_plugin(), class = "skip")
    local_default_device("cpu")
    expect_no_condition(skip_if_no_pjrt_plugin(), class = "skip")
  })
})

describe("eager code", {
  it("rejects an array of another backend instead of guessing a default", {
    skip_if_no_quickr()
    x <- with_backend("quickr", nv_array(1L))
    expect_error(x + 1.5, "quickr")
    expect_error(nv_fill_like(x, 0), "belongs to the .*quickr.* backend")
  })
})

describe("the plain backend", {
  it("converts a bit64::integer64 payload by value", {
    out <- jit(function() nv_array(bit64::as.integer64(c(3, 5)), dtype = "i64"))()
    expect_equal(as.integer(as_array(out)), c(3L, 5L))
  })
})
