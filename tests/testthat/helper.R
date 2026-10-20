# The platform of the device the suite runs on, from `ANVL_DEFAULT_DEVICE`.
test_platform <- function() {
  sub(":.*$", "", Sys.getenv("ANVL_DEFAULT_DEVICE", "cpu"))
}

is_cuda <- function() {
  test_platform() == "cuda"
}

# Set the fallback anvl read from `ANVL_DEFAULT_<name>` when it was loaded
# (`"dtypes"` or `"device"`) for the calling scope.
local_env_default <- function(name, value, envir = parent.frame()) {
  key <- paste0("env_default_", name)
  old <- globals[[key]]
  globals[[key]] <- value
  withr::defer(assign(key, old, envir = globals), envir = envir)
}

# Clear the `anvl.default_dtypes` / `ANVL_DEFAULT_DTYPES` override for the
# calling scope, so that the defaults are whatever the active backend registers. For the few tests that
# assert the *registered* pair and would otherwise see the suite-wide override
# `ANVL_DEFAULT_DTYPES` sets (see `setup.R`).
local_registered_default_dtypes <- function(envir = parent.frame()) {
  withr::local_options(list(anvl.default_dtypes = NULL), .local_envir = envir)
  local_env_default("dtypes", NULL, envir)
}

is_cpu <- function() {
  test_platform() == "cpu"
}

# Clear the `anvl.default_device` / `ANVL_DEFAULT_DEVICE` override for the
# calling scope, so that the default device is the first CPU device. For the few tests that assert that and
# would otherwise see the suite-wide override `ANVL_DEFAULT_DEVICE`
# sets (see `setup.R`).
local_unset_default_device <- function(envir = parent.frame()) {
  withr::local_options(list(anvl.default_device = NULL), .local_envir = envir)
  local_env_default("device", NULL, envir)
}

if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "torch-helpers.R", package = "anvl"), local = TRUE)
}

# Sampler factories used to constrain the input domain when comparing primitives
# against torch (e.g. `acos` only accepts values in (-1, 1)).  Each returns a
# `function(shp, dtype)` matching the shape expected by `expect_jit_torch_unary()`
# / `verify_grad_uni()`.
# jarl-ignore unused_function: used from inst/extra-tests
sampler_unif <- function(lower, upper) {
  function(shp, dtype) {
    n <- if (length(shp)) prod(shp) else 1L
    vals <- runif(n, lower, upper)
    if (length(shp)) array(vals, shp) else vals
  }
}

# jarl-ignore unused_function: used from inst/extra-tests
sampler_rnorm <- function(mean = 0, sd = 1) {
  function(shp, dtype) {
    n <- if (length(shp)) prod(shp) else 1L
    vals <- rnorm(n, mean, sd)
    if (length(shp)) array(vals, shp) else vals
  }
}

# Sample values bounded away from zero (|v| >= min_abs). Useful when a
# primitive's gradient or domain blows up at 0, e.g. division, `atan2` at the
# origin, or `reduce_prod`'s per-element gradient.
# jarl-ignore unused_function: used from inst/extra-tests
sampler_nonzero <- function(min_abs = 0.5) {
  function(shp, dtype) {
    n <- if (length(shp)) prod(shp) else 1L
    vals <- rnorm(n)
    small <- abs(vals) < min_abs
    sgn <- ifelse(vals[small] >= 0, 1, -1)
    vals[small] <- sgn * (abs(vals[small]) + min_abs)
    if (length(shp) == 0L) vals else array(vals, shp)
  }
}

verify_zero_grad_unary <- function(prim_fn, x, f_wrapper = NULL) {
  # We can only take gradients w.r.t. float arrays, so the outer input is f32
  # and the actual dtype is restored inside the function before calling prim_fn.
  x_dtype <- dtype(x)
  x_f32 <- nv_convert(x, "f32")
  if (is.null(f_wrapper)) {
    f <- function(x) {
      x_inner <- nv_convert(x, x_dtype)
      out <- prim_fn(x_inner)
      out <- nv_convert(out, "f32")
      nv_sum(out, axes = 1L, drop = TRUE)
    }
  } else {
    f <- f_wrapper
  }
  grads <- jit(gradient(f))(x_f32)
  expected <- nv_array(0, shape = shape(x), dtype = "f32")
  testthat::expect_equal(grads[[1L]], expected)
}

verify_zero_grad_binary <- function(prim_fn, x, y) {
  x_dtype <- dtype(x)
  y_dtype <- dtype(y)
  x_f32 <- nv_convert(x, "f32")
  y_f32 <- nv_convert(y, "f32")
  f <- function(x, y) {
    x_inner <- nv_convert(x, x_dtype)
    y_inner <- nv_convert(y, y_dtype)
    out <- prim_fn(x_inner, y_inner)
    out <- nv_convert(out, "f32")
    nv_sum(out, axes = 1L, drop = TRUE)
  }
  grads <- jit(gradient(f))(x_f32, y_f32)
  expected1 <- nv_array(0, shape = shape(x), dtype = "f32")
  expected2 <- nv_array(0, shape = shape(y), dtype = "f32")
  testthat::expect_equal(grads[[1L]], expected1)
  testthat::expect_equal(grads[[2L]], expected2)
}

# A side effect counting the calls of the primitive `tick` it returns. Its
# value starts at `start` in the program (`from = "program"`) or comes from
# `store$value` on the host (`from = "host"`); with `received = TRUE` the host
# receives the value a program leaves behind in `store$received`, otherwise the
# program drops it. Both are removed when `envir` exits.
local_counter <- function(from = "program", start = 0L, received = TRUE, envir = parent.frame()) {
  store <- new.env(parent = emptyenv())
  store$value <- start
  register_side_effect(
    "counter",
    avals = list(counter = AbstractArray(dtype = "i32", shape = integer())),
    get = if (from == "host") function() list(store$value),
    build = if (from == "program") {
      function(desc) {
        list(
          counter = graph_desc_add(
            prim_fill,
            list(),
            params = list(value = start, dtype = "i32", shape = integer()),
            infer_fn = infer_fill,
            desc = desc
          )[[1L]]
        )
      }
    },
    after_run = if (received) {
      function(values) {
        store$received <- as_array(values[[1L]])
      }
    }
  )
  withr::defer(
    {
      globals$side_effects$counter <- NULL
    },
    envir = envir
  )

  tick <- new_primitive(
    "tick",
    function() {
      graph_desc_add(self, list(), infer_fn = function(counter) list(value = counter, counter = counter))[[1L]]
    },
    side_effects = "counter",
    register = FALSE
  )
  tick[["stablehlo"]] <- function(counter) {
    list(counter, hlo_add(counter, hlo_scalar(1L, dtype = "i32", func = counter$func)))
  }
  list(store = store, tick = tick)
}
