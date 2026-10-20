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

# The 32-bit words of `x`, a `ui32` or `ui64` array, low word first.
u32_words <- function(x) {
  if (dtype(x) == "ui64") {
    x <- nv_bitcast_convert(x, dtype = "ui32")
  }
  as.numeric(as_array(x))
}

# The four 32-bit words of the random state `random_state`, by default the
# global one: those of its key, then of its counter, low word first.
state_words <- function(random_state = nv_get_random_state()) {
  c(u32_words(random_state$random_key), u32_words(random_state$random_counter))
}

# The random state with the four 32-bit words `words`, as `state_words()` gives
# them.
random_state_of_words <- function(words) {
  structure(
    list(
      random_key = nv_array(words[1:2], dtype = "ui32"),
      random_counter = nv_bitcast_convert(nv_array(words[3:4], dtype = "ui32"), dtype = "ui64")
    ),
    class = "AnvlRandomState"
  )
}
