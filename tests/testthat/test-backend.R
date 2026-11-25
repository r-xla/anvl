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

describe("skip_if_no_plugin", {
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
    expect_condition(skip_if_no_plugin("cpu"), class = "skip")
    withr::local_envvar(PJRT_INSTALL = "0")
    expect_condition(skip_if_no_plugin("cpu"), class = "skip")
  })

  it("does not skip when the plugin is downloaded", {
    local_no_pjrt_plugin()
    dir.create(file.path(tools::R_user_dir("pjrt", "cache"), "cpu"), recursive = TRUE)
    expect_no_condition(skip_if_no_plugin("cpu"), class = "skip")
  })

  it("does not skip when PJRT_PLUGIN_PATH_<PLATFORM> points to the plugin", {
    local_no_pjrt_plugin()
    withr::local_envvar(PJRT_PLUGIN_PATH_CPU = "libpjrt_cpu.so")
    expect_no_condition(skip_if_no_plugin("cpu"), class = "skip")
    expect_condition(skip_if_no_plugin("cuda"), class = "skip")
  })

  it("does not skip when PJRT_INSTALL=1 allows downloading the plugin", {
    local_no_pjrt_plugin()
    withr::local_envvar(PJRT_INSTALL = "1")
    expect_no_condition(skip_if_no_plugin("cpu"), class = "skip")
  })

  it("defaults to the platform of the default device", {
    local_no_pjrt_plugin()
    withr::local_envvar(PJRT_PLUGIN_PATH_CPU = "libpjrt_cpu.so")
    local_default_device("cuda:1")
    expect_condition(skip_if_no_plugin(), class = "skip")
    local_default_device("cpu")
    expect_no_condition(skip_if_no_plugin(), class = "skip")
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

describe("register_backend", {
  # A backend built from exported functions only, the way a package would build
  # one: it holds arrays as R arrays and runs a jitted function by tracing it
  # and evaluating each statement with the primitive's `"toy"` rule.
  toy_device <- function() structure("cpu", class = c("ToyDevice", "AnvlDevice"))

  toy_array <- function(data, dtype, shape) {
    mode <- switch(substr(as.character(dtype), 1L, 1L), f = "double", b = "logical", "integer")
    data <- as.vector(data, mode)
    if (length(shape)) {
      data <- array(data, dim = shape)
    }
    structure(
      list(data = data, dtype = as_dtype(dtype), shape = as.integer(shape), backend = "toy"),
      class = "AnvlArray"
    )
  }

  toy_run <- function(f, args) {
    graph <- trace_fn(f, args = args)
    values <- utils::hashtab()
    value_of <- function(node) {
      if (inherits(node, "GraphLiteral")) node$aval$data else utils::gethash(values, node)
    }
    inputs <- Filter(function(x) inherits(x, "AnvlArray"), flatten(args))
    Map(function(node, x) utils::sethash(values, node, as_array(x)), graph$inputs, inputs)
    for (statement in graph$statements) {
      outputs <- do.call(statement$primitive[["toy"]], c(lapply(statement$inputs, value_of), statement$params))
      Map(function(node, x) utils::sethash(values, node, x), statement$outputs, outputs)
    }
    outputs <- lapply(graph$outputs, function(node) {
      nv_array(value_of(node), dtype = dtype(node$aval), shape = shape(node$aval))
    })
    unflatten(graph$out_tree, outputs)
  }

  toy_backend <- function() {
    AnvlBackend(
      new_data = function(data, dtype, shape, device, row_major = FALSE, check = NULL) toy_array(data, dtype, shape),
      new_empty = function(dtype, shape, device) toy_array(0, dtype, shape),
      dtype = function(x) x$dtype,
      shape = function(x) x$shape,
      as_array = function(x, check) x$data,
      as_raw = function(x, row_major) stop("The toy backend has no raw payloads."),
      platform = function(x) "cpu",
      device = function(x) toy_device(),
      new_device = function(x) toy_device(),
      print_data = function(x, footer, ...) {
        print(x$data, ...)
        cat(footer, "\n")
      },
      jit = function(f, static, cache_size, device = NULL) {
        function() toy_run(f, lapply(as.list(match.call())[-1L], eval, envir = parent.frame()))
      },
      await_data = function(x) invisible(NULL),
      default_dtypes = list(float = "f64", int = "i32")
    )
  }

  local_toy_backend <- function(envir = parent.frame()) {
    register_backend("toy", toy_backend())
    register_rule_type("toy")
    registerS3method("backend", "ToyDevice", function(x, ...) "toy")
    prim_add[["toy"]] <- function(lhs, rhs) list(lhs + rhs)
    prim_mul[["toy"]] <- function(lhs, rhs) list(lhs * rhs)
    withr::defer(
      {
        prim_add[["toy"]] <- NULL
        prim_mul[["toy"]] <- NULL
        globals$interpretation_rules <- setdiff(globals$interpretation_rules, "toy")
        rm("toy", envir = globals$backends)
      },
      envir = envir
    )
    local_backend("toy", envir = envir)
  }

  it("makes a backend that a package defines selectable", {
    local_registered_default_dtypes()
    local_toy_backend()
    expect_equal(active_backend(), "toy")
    x <- nv_array(c(1, 2))
    expect_equal(backend(x), "toy")
    expect_equal(dtype(x), as_dtype("f64"))
    expect_true(is_device(device(x)))
    expect_equal(backend(nv_device("cpu")), "toy")
  })

  it("runs jitted and eager code with the rules attached for the backend", {
    local_toy_backend()
    f <- jit(function(x, y) x * y + x)
    out <- f(nv_array(c(1, 2)), nv_array(c(3, 4)))
    expect_equal(backend(out), "toy")
    expect_equal(as.vector(as_array(out)), c(4, 10))
    expect_equal(as.vector(as_array(nv_array(c(1, 2)) * 2)), c(2, 4))
  })

  it("checks that the backend is an AnvlBackend when it is first used", {
    register_backend("not_a_backend", list())
    withr::defer(rm("not_a_backend", envir = globals$backends))
    local_backend("not_a_backend")
    expect_error(nv_array(1), "AnvlBackend")
  })

  it("rejects a name that the option anvl.default_dtypes reserves", {
    expect_error(register_backend("float", toy_backend()), "data type category")
  })
})
