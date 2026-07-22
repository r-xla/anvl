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

describe("resolve_backends", {
  it("defaults to the pjrt backend", {
    expect_equal(resolve_backends("pjrt"), list(active = "pjrt", default = "pjrt"))
  })

  it("uses a single active backend as the default", {
    expect_equal(resolve_backends("quickr"), list(active = "quickr", default = "quickr"))
  })

  it("uses the first active backend as the default when several are active", {
    expect_equal(
      resolve_backends(c("quickr", "pjrt")),
      list(active = c("quickr", "pjrt"), default = "quickr")
    )
  })

  it("drops duplicates", {
    expect_equal(resolve_backends(c("pjrt", "pjrt")), list(active = "pjrt", default = "pjrt"))
  })

  it("errors on unknown backend names", {
    expect_error(resolve_backends(c("pjrt", "torch")), "unknown backend \"torch\"")
  })

  it("errors on non-character or empty values", {
    expect_error(resolve_backends(character()), "non-empty character vector")
    expect_error(resolve_backends(1L), "non-empty character vector")
    expect_error(resolve_backends(NA_character_), "non-empty character vector")
  })
})

describe("inactive backends", {
  # The test suite activates both backends (see setup.R), so re-activate only
  # some of them to observe what a user with the default configuration sees.
  local_active <- function(backends, envir = parent.frame()) {
    old <- globals$active_backends
    restore <- function() {
      activate_backends(old)
    }
    withr::defer(restore(), envir = envir)
    activate_backends(backends)
  }

  it("local_backend() errors and points at the option", {
    local_active("pjrt")
    expect_error(local_backend("quickr"), "Backend \"quickr\" is not active")
    expect_error(local_backend("quickr"), "anvl.backends")
  })

  it("with_backend() errors", {
    local_active("pjrt")
    expect_error(with_backend("quickr", 1), "is not active")
  })

  it("jit() errors", {
    local_active("pjrt")
    expect_error(jit(identity, backend = "quickr"), "is not active")
  })

  it("nv_array() errors", {
    local_active("pjrt")
    expect_error(nv_array(1, backend = "quickr"), "is not active")
  })

  it("an entirely unknown backend is rejected", {
    expect_error(local_backend("torch"), "Unknown backend \"torch\"")
  })
})

# The suite inherits the default `anvl.backends`. A developer who overrides that
# option in their `.Rprofile` sees this fail first, naming the reason, rather
# than a pile of confusing backend errors further down.
test_that("the test suite runs with the default backends", {
  expect_equal(globals$active_backends, c("pjrt", "quickr"))
  expect_equal(default_backend(), "pjrt")
})

describe("the default backend is always concrete", {
  it("is the first active backend, not 'auto'", {
    skip_if_no_quickr()
    expect_equal(globals$active_backends[[1L]], "pjrt")
    expect_equal(default_backend(), "pjrt")
    expect_equal(backend(nv_array(1)), "pjrt")
  })

  it("'auto' is rejected as a default backend", {
    expect_error(local_backend("auto"), "not a backend")
    expect_error(with_backend("auto", 1), "backend = \"auto\"")
    expect_error(nv_array(1, backend = "auto"), "not a backend")
  })
})

describe("jit() with backend = NULL", {
  it("infers the backend per call while several backends are active", {
    skip_if_no_quickr()
    f <- jit(identity)
    expect_equal(backend(f), "auto")
    expect_equal(backend(f(nv_array(1, backend = "quickr"))), "quickr")
    expect_equal(backend(f(nv_array(1, backend = "pjrt"))), "pjrt")
  })

  it("falls back to the default backend when no input names one", {
    skip_if_no_quickr()
    expect_equal(backend(jit(identity)(1)), "pjrt")
    local_backend("quickr")
    expect_equal(backend(jit(identity)(1)), "quickr")
  })

  it("pins the only active backend when just one is active", {
    old <- globals$active_backends
    withr::defer(activate_backends(old))
    activate_backends("pjrt")
    f <- jit(identity)
    expect_equal(backend(f), "pjrt")
  })
})
