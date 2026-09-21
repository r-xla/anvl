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
  ok <- test_integerish(x, lower = 0L, min.len = min_len, any.missing = FALSE, null.ok = FALSE)
  fmt <- function(x) {
    sprintf("(%s)", paste0(x, collapse = ", "))
  }
  if (!isTRUE(ok)) {
    if (is.null(x) || !is.numeric(x)) {
      cli_abort("{.arg {var_name}} must be an integer vector, not {.cls {class(x)}}")
    }
    if (anyNA(x)) {
      cli_abort(c(
        "{.arg {var_name}} must not contain missing values",
        x = "Got {fmt(x)}."
      ))
    }
    if (length(x) < min_len) {
      cli_abort(c(
        "{.arg {var_name}} must have at least {min_len} element{?s}",
        x = "Got {fmt(x)}."
      ))
    }
    if (any(x < 0)) {
      cli_abort(c(
        "{.arg {var_name}} must not contain a negative axis size.",
        x = "Got {fmt(x)}."
      ))
    }
    cli_abort(c(
      "{.arg {var_name}} must contain whole numbers in the integer range",
      x = "Got {fmt(x)}."
    ))
  }
  as.integer(x)
}

# Normalize possibly-negative axis indices.
#
# Negative values count from the end: `-1` is the last axis, `-2` the
# second-to-last. `max_axis` is the largest admissible axis. It is the rank
# of the array for most operations, but `rank + 1` for operations that insert a
# new axis (e.g. `nv_unsqueeze()`).
# Returns the resolved (positive) axes as an integer vector.
resolve_axes <- function(axes, max_axis, arg = rlang::caller_arg(axes), unique = FALSE) {
  if (!test_integerish(axes, any.missing = FALSE, null.ok = FALSE)) {
    cli_abort("{.arg {arg}} must be an integer vector without missing values, not {.cls {class(axes)}}")
  }
  original <- as.integer(axes)
  resolved <- original
  negative <- original < 0L
  resolved[negative] <- max_axis + 1L + resolved[negative]
  invalid <- resolved < 1L | resolved > max_axis
  if (any(invalid)) {
    if (max_axis < 1L) {
      cli_abort(c(
        "{.arg {arg}} cannot be used, there is no axis to select.",
        x = "Got {.val {original[invalid]}}."
      ))
    }
    cli_abort(c(
      "{.arg {arg}} must be between 1 and {max_axis}, or between {-max_axis} and -1 to count from the end.",
      x = "Got {.val {original[invalid]}}."
    ))
  }
  if (unique && anyDuplicated(resolved)) {
    cli_abort(c(
      "{.arg {arg}} must not contain duplicate axes.",
      x = "Got {.val {original}}."
    ))
  }
  resolved
}

# Like `resolve_axes()`, but for a single axis.
resolve_axis <- function(axis, max_axis, arg = rlang::caller_arg(axis)) {
  if (length(axis) != 1L) {
    cli_abort("{.arg {arg}} must have length 1, not {length(axis)}")
  }
  resolve_axes(axis, max_axis, arg = arg)
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
      "{.arg {arg}} must contain only non-negative values, or {.val {-1L}} to infer an axis size.",
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
      "Cannot infer the size of axis {inferred} of {.arg {arg}}.",
      # The `-1` is the axis being asked for, so it is shown as `?` rather
      # than as a size.
      x = "{nelts} element{?s} cannot be divided evenly into shape {shape_repr(replace(shape, inferred, '?'))}." # nolint
    ))
  }
  shape[inferred] <- as.integer(nelts / known)
  shape
}

# Like `assert_float_dtype()`, but only the two widths the RNG is written for:
# it assembles floats out of random bits, so it needs a 32- or 64-bit layout
# and cannot serve `bf16` or `f16` even though those are float data types.
# Returns the converted DataType.
assert_rng_float_dtype <- function(x, arg = rlang::caller_arg(x), hint = NULL) {
  dt <- as_dtype(x)
  if (!is_dtype_float(dt)) {
    cli_abort(c(
      "{.arg {arg}} must be a float data type.",
      "x" = "Got {.val {as.character(dt)}}.",
      "i" = hint
    ))
  }
  if (!dtype_width(dt) %in% c(32L, 64L)) {
    cli_abort(c(
      "{.arg {arg}} must be a 32- or 64-bit float data type.",
      "x" = "Got {.val {as.character(dt)}}.",
      "i" = hint
    ))
  }
  dt
}

assert_fill_value <- function(value, dtype, arg = rlang::caller_arg(value)) {
  dt <- as_dtype(dtype)
  is_int64 <- inherits(value, "integer64")
  is_number <- is.numeric(value) || is_int64

  if (length(value) != 1L) {
    cli_abort(c(
      "{.arg {arg}} must be a scalar.",
      "x" = "Got {.obj_type_friendly {value}} of length {length(value)}."
    ))
  }
  if (is.na(value) && !is.nan(value)) {
    cli_abort(c(
      "{.arg {arg}} must not be {.val {NA}}.",
      "i" = "There is no missing value at the XLA level; {.val {NaN}} is the closest a float comes."
    ))
  }

  # What an integer data type needs is a whole *number*, not an R integer:
  # `1` and `1L` both build at `i32`, while `1.5` builds at neither. This keeps
  # the fills that do not know their data type statically (`zeros()`, `ones()`,
  # `nv_eye()`, `nv_diag()`, the gradient zeroing) free to write a plain `0`.
  # `test_int()` only accepts a double that fits into an R integer, so the
  # out-of-range whole doubles are recognized here and rejected below.
  is_whole <- is_int64 ||
    test_int(value) ||
    (is.double(value) && is.finite(value) && value == trunc(value))
  # A whole double is built as an R integer, so one beyond that range would
  # silently arrive at the backend as `NA`.
  too_large <- is_whole && !is_int64 && !is.integer(value) && abs(value) > .Machine$integer.max
  if (too_large && !is_dtype_float(dt)) {
    int_max <- .Machine$integer.max
    cli_abort(c(
      "{.arg {arg}} must be no larger than {.val {int_max}} to be built at data type {.val {as.character(dt)}}.", # nolint
      "x" = "Got {.val {value}}.",
      "i" = "A whole number is built as an R integer."
    ))
  }

  ok <- if (is_dtype_bool(dt)) {
    is.logical(value) || (is_whole && (value == 0 || value == 1))
  } else if (is_dtype_uint(dt)) {
    is_whole && value >= 0
  } else if (is_dtype_int(dt)) {
    is_whole
  } else {
    is_number
  }
  if (ok) {
    return(invisible(value))
  }

  wanted <- if (is_dtype_bool(dt)) {
    cli::format_inline("a logical, or {.code 0} or {.code 1}")
  } else if (is_dtype_uint(dt)) {
    "a non-negative whole number"
  } else if (is_dtype_int(dt)) {
    "a whole number"
  } else {
    "a number"
  }
  cli_abort(c(
    "{.arg {arg}} must be {wanted} to be built at data type {.val {as.character(dt)}}.",
    "x" = "Got {.obj_type_friendly {value}}{if (is_number) cli::format_inline(' {.val {value}}') else ''}."
  ))
}

# Convert `x` to a DataType via `as_dtype()` and assert it is numeric in the
# sense `?dtypes` gives the word: integer or float, but not `bool`. Returns the
# converted DataType.
assert_numeric_dtype <- function(x, arg = rlang::caller_arg(x), hint = NULL) {
  dt <- as_dtype(x)
  if (is_dtype_bool(dt)) {
    cli_abort(c(
      "{.arg {arg}} must be a numeric data type.",
      "x" = "Got {.val {as.character(dt)}}, which is boolean.",
      "i" = hint
    ))
  }
  dt
}

# Convert `x` to a DataType via `as_dtype()` and assert it belongs to the float
# category. Returns the converted DataType.
assert_float_dtype <- function(x, arg = rlang::caller_arg(x), hint = NULL) {
  dt <- as_dtype(x)
  # The float category, as `?dtypes` defines it, so this and `is_dtype_float()`
  # agree on what counts as a float.
  if (!is_dtype_float(dt)) {
    cli_abort(c(
      "{.arg {arg}} must be a float data type.",
      "x" = "Got {.val {as.character(dt)}}.",
      "i" = hint
    ))
  }
  dt
}

assert_some_arrays <- function(..., call = rlang::caller_env()) {
  if (...length() == 0L) {
    cli_abort(
      "At least one array is required, but none was given.",
      call = call
    )
  }
  invisible(NULL)
}

# `batched = TRUE` accepts leading batch axes and checks only the last two,
# which is what the operations whose lowering broadcasts over batches take
# (`prim_chol()`, `prim_triangular_solve()`); the others are strictly 2-D.
assert_linalg_matrix <- function(x, arg, square = FALSE, batched = FALSE) {
  s <- shape(x)
  if (batched) {
    if (length(s) < 2L) {
      cli_abort(c(
        "{.arg {arg}} must have at least 2 axes, the last two forming a matrix.",
        "x" = "Got shape {shape_repr(s)}."
      ))
    }
  } else if (length(s) != 2L) {
    cli_abort(c(
      "{.arg {arg}} must be a 2-D matrix.",
      "x" = "Got shape {shape_repr(s)}."
    ))
  }
  if (any(s == 0L)) {
    cli_abort(c(
      "{.arg {arg}} must not have any zero-sized axis.",
      "x" = "Got shape {shape_repr(s)}."
    ))
  }
  mat <- utils::tail(s, 2L)
  if (square && mat[[1L]] != mat[[2L]]) {
    cli_abort(c(
      "{.arg {arg}} must be square in its last two axes.",
      "x" = "Got shape {shape_repr(s)}."
    ))
  }
  if (!is_dtype_float(peek_dtype(x))) {
    cli_abort(c(
      "{.arg {arg}} must have a float data type.",
      "x" = "Got dtype {.val {as.character(peek_dtype(x))}}."
    ))
  }
  invisible(NULL)
}

# Assert that `x` is a boolean array, or an R value that would become one.
# `hint` is a character vector, each element shown as its own bullet.
assert_boolean_array <- function(x, arg = rlang::caller_arg(x), hint = NULL) {
  dt <- peek_dtype(x)
  if (!is_dtype_bool(dt)) {
    cli_abort(c(
      "{.arg {arg}} must be a boolean array.",
      "x" = "Got data type {.val {as.character(dt)}}.",
      info_bullets(hint)
    ))
  }
  x
}

# Name each element "i" so cli shows it as its own info bullet. Naming a
# multi-element vector as a whole would renumber the names to "i1", "i2", ...
# and lose the bullets.
info_bullets <- function(x) {
  if (!length(x)) {
    return(NULL)
  }
  names(x) <- rep("i", length(x))
  x
}
