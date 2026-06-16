is_cuda <- function() {
  Sys.getenv("PJRT_PLATFORM") == "cuda"
}

is_cpu <- function() {
  Sys.getenv("PJRT_PLATFORM", "cpu") == "cpu"
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
      nv_reduce_sum(out, dims = 1L, drop = TRUE)
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
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  }
  grads <- jit(gradient(f))(x_f32, y_f32)
  expected1 <- nv_array(0, shape = shape(x), dtype = "f32")
  expected2 <- nv_array(0, shape = shape(y), dtype = "f32")
  testthat::expect_equal(grads[[1L]], expected1)
  testthat::expect_equal(grads[[2L]], expected2)
}
