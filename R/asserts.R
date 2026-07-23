#' @title Assert Shape Vector
#' @description
#' Check whether an input is a valid shape vector (integer vector with all positive values).
#' @param x Object to check.
#' @param min_len (`integer(1)`)\cr
#'   Minimum length of the shape vector. Default is 1.
#' @param var_name (`character(1)`)\cr
#'   Name of the variable to use in error messages.
#' @return Invisibly returns `x` if the assertion passes.
#' @keywords internal
assert_shapevec <- function(x, min_len = 0L, var_name = rlang::caller_arg(x)) {
  ok <- test_integerish(x, lower = 1, min.len = min_len, any.missing = FALSE, null.ok = FALSE)
  if (!isTRUE(ok)) {
    if (is.null(x) || !is.numeric(x)) {
      cli_abort("{.arg {var_name}} must be an integer vector, not {.cls {class(x)}}")
    }
    if (anyNA(x)) {
      cli_abort("{.arg {var_name}} must not contain missing values")
    }
    if (length(x) < min_len) {
      cli_abort("{.arg {var_name}} must have at least {min_len} element{?s}")
    }
    if (any(x < 1)) {
      cli_abort("{.arg {var_name}} must contain only positive integers (>= 1)")
    }
  }
  as.integer(x)
}

# Normalize possibly-negative dimension indices.
#
# Negative values count from the end: `-1` is the last dimension, `-2` the
# second-to-last. `max_dim` is the largest admissible dimension. It is the rank
# of the array for most operations, but `rank + 1` for operations that insert a
# new dimension (e.g. `nv_unsqueeze()`).
# Returns the resolved (positive) dimensions as an integer vector.
resolve_dims <- function(dims, max_dim, arg = rlang::caller_arg(dims), unique = FALSE) {
  if (!test_integerish(dims, any.missing = FALSE, null.ok = FALSE)) {
    cli_abort("{.arg {arg}} must be an integer vector without missing values, not {.cls {class(dims)}}")
  }
  original <- as.integer(dims)
  resolved <- original
  negative <- original < 0L
  resolved[negative] <- max_dim + 1L + resolved[negative]
  invalid <- resolved < 1L | resolved > max_dim
  if (any(invalid)) {
    if (max_dim < 1L) {
      cli_abort(c(
        "{.arg {arg}} cannot be used, there is no dimension to select.",
        x = "Got {.val {original[invalid]}}."
      ))
    }
    cli_abort(c(
      "{.arg {arg}} must be between 1 and {max_dim}, or between {-max_dim} and -1 to count from the end.",
      x = "Got {.val {original[invalid]}}."
    ))
  }
  if (unique && anyDuplicated(resolved)) {
    cli_abort(c(
      "{.arg {arg}} must not contain duplicate dimensions.",
      x = "Got {.val {original}}."
    ))
  }
  resolved
}

# Like `resolve_dims()`, but for a single dimension.
resolve_dim <- function(dim, max_dim, arg = rlang::caller_arg(dim)) {
  if (length(dim) != 1L) {
    cli_abort("{.arg {arg}} must have length 1, not {length(dim)}")
  }
  resolve_dims(dim, max_dim, arg = arg)
}

# Resolve a `-1` placeholder in a reshape target shape by inferring the
# corresponding extent from the total number of elements `nelts`.
# Returns the resolved shape as an integer vector.
resolve_reshape_shape <- function(shape, nelts, arg = rlang::caller_arg(shape)) {
  if (!test_integerish(shape, any.missing = FALSE, null.ok = FALSE)) {
    cli_abort("{.arg {arg}} must be an integer vector without missing values, not {.cls {class(shape)}}")
  }
  shape <- as.integer(shape)
  invalid <- shape < -1L
  if (any(invalid)) {
    cli_abort(c(
      "{.arg {arg}} must contain only non-negative values, or {.val {-1L}} to infer a dimension.",
      x = "Got {.val {shape[invalid]}}."
    ))
  }
  inferred <- which(shape == -1L)
  if (length(inferred) == 0L) {
    return(shape)
  }
  if (length(inferred) > 1L) {
    cli_abort(c(
      "{.arg {arg}} must contain at most one {.val {-1L}}.",
      x = "Got {length(inferred)} at positions {.val {inferred}}."
    ))
  }
  known <- prod(shape[-inferred])
  if (known <= 0 || nelts %% known != 0) {
    cli_abort(c(
      "Cannot infer dimension {inferred} of {.arg {arg}}.",
      x = "{nelts} element{?s} cannot be divided evenly into shape {.val {shape}}."
    ))
  }
  shape[inferred] <- as.integer(nelts / known)
  shape
}

# Convert `x` to a DataType via `as_dtype()` and assert it is a floating-point
# dtype (f32 or f64). Returns the converted DataType.
assert_float_dtype <- function(x, arg = rlang::caller_arg(x)) {
  dt <- as_dtype(x)
  # Deliberately narrower than is_dtype_float(): the callers (rng, sampling)
  # assume 32/64-bit float layouts.
  if (dt != "f32" && dt != "f64") {
    cli_abort(c(
      "{.arg {arg}} must be a floating-point dtype (f32 or f64).",
      "x" = "Got {.val {as.character(dt)}}."
    ))
  }
  dt
}

assert_linalg_matrix <- function(x, arg, square = FALSE) {
  s <- shape(x)
  if (length(s) != 2L) {
    cli_abort(c(
      "{.arg {arg}} must be a 2-D matrix.",
      "x" = "Got shape {xlamisc::shapevec_repr(s)}."
    ))
  }
  if (any(s == 0L)) {
    cli_abort(c(
      "{.arg {arg}} must not have any zero-sized dimension.",
      "x" = "Got shape {xlamisc::shapevec_repr(s)}."
    ))
  }
  if (square && s[[1L]] != s[[2L]]) {
    cli_abort(c(
      "{.arg {arg}} must be a square matrix.",
      "x" = "Got shape {xlamisc::shapevec_repr(s)}."
    ))
  }
  if (!is_dtype_float(dtype(x))) {
    cli_abort(c(
      "{.arg {arg}} must have a floating-point dtype.",
      "x" = "Got dtype {.val {as.character(dtype(x))}}."
    ))
  }
  invisible(NULL)
}
