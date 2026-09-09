#' @title Assert Shape Vector
#' @description
#' Check whether an input is a valid shape vector: whole, non-negative axis
#' sizes. A zero-size axis is legal, so 0 is accepted.
#' @param x Object to check.
#' @param min_len (`integer(1)`)\cr
#'   Minimum number of axes. Default is 0, which admits `integer()` -- the
#'   shape of a scalar.
#' @param var_name (`character(1)`)\cr
#'   Name of the variable to use in error messages.
#' @return (`integer()`)\cr
#'   `x` as an integer vector.
#' @keywords internal
assert_shapevec <- function(x, min_len = 0L, var_name = rlang::caller_arg(x)) {
  # `lower = 0`: a zero-size axis is a legal shape, and the constructors
  # (`nv_fill()`, `nv_iota()`, `nv_empty()`) all accept one.
  ok <- test_integerish(
    x,
    lower = 0,
    upper = .Machine$integer.max,
    min.len = min_len,
    any.missing = FALSE,
    null.ok = FALSE
  )
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
    if (any(x < 0)) {
      cli_abort(c(
        "{.arg {var_name}} must not contain a negative axis size.",
        x = "Got {.val {x}}."
      ))
    }
    # Everything else `test_integerish()` rejects: a fractional size, or one
    # above what an R integer holds. Without this arm both fell through to
    # `as.integer()`, which truncates the first and turns the second into `NA`.
    int_max <- .Machine$integer.max
    cli_abort(c(
      "{.arg {var_name}} must contain whole numbers no larger than {.val {int_max}}.",
      x = "Got {.val {x}}."
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

# Like `assert_float_dtype()`, but only the two widths written for: the RNG
# assembles floats out of random bits, so it needs a 32- or 64-bit layout, and
# `nv_pnorm()` / `nv_qnorm()` carry one coefficient set per width. Neither can
# serve `bf16` or `f16` even though those are floats.
assert_rng_float_dtype <- function(x, arg = rlang::caller_arg(x), hint = NULL) {
  dt <- assert_float_dtype(x, arg = arg, hint = hint)
  if (!dtype_width(dt) %in% c(32L, 64L)) {
    cli_abort(c(
      "{.arg {arg}} must be a 32- or 64-bit float data type.",
      "x" = "Got {.val {as.character(dt)}}.",
      "i" = hint
    ))
  }
  dt
}

# The R value a `prim_fill()` / `nv_fill()` call builds at `dtype` has to be
# something that data type can hold: a number for a float, a whole number for an
# integer, a non-negative whole number for an unsigned one and a logical for
# `bool`. This is a check on the *R value*, not on the range of the data type --
# `prim_fill(300L, dtype = "i8")` is still the backend's business.
assert_fill_value <- function(value, dtype, arg = rlang::caller_arg(value)) {
  dt <- as_dtype(dtype)
  is_int64 <- inherits(value, "integer64")
  # `integer64` is a double under the hood, so it is counted in explicitly.
  is_number <- (is.numeric(value) && !is.logical(value)) || is_int64
  # `trunc()` on an `integer64` would drop the class, and it is whole anyway.
  is_whole <- is_int64 || (is_number && is.finite(value) && value == trunc(value))

  # XLA has no missing value, and an `NA` otherwise reaches the backend and
  # fails there with a raw MLIR message.
  if (length(value) == 1L && !is_int64 && is.na(value) && !is.nan(value)) {
    cli_abort(c(
      "{.arg {arg}} must not be {.val {NA}}.",
      "i" = "There is no missing value at the XLA level; {.val {NaN}} is the closest a float comes."
    ))
  }

  ok <- if (is_dtype_bool(dt)) {
    is.logical(value)
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
    "a logical"
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

# Assert that the start indices of a dynamic slice are one integer scalar per
# axis of `x`. These primitives build their output aval themselves rather than
# calling stablehlo's inference, so a mistake here survives to the PJRT
# compiler and reaches the user as a raw MLIR dump.
assert_start_indices <- function(x, indices, call = rlang::caller_env()) {
  if (length(indices) != naxes(x)) {
    cli_abort(
      c(
        "One start index per axis of {.arg x} is required.",
        x = "{.arg x} has {naxes(x)} ax{?is/es}, but {length(indices)} ind{?ex/ices} {?was/were} given."
      ),
      call = call
    )
  }
  for (i in seq_along(indices)) {
    idx <- indices[[i]]
    if (naxes(idx) != 0L) {
      cli_abort(
        c(
          "Every start index must be a scalar.",
          x = "Index {i} has shape {shape_repr(shape(idx))}."
        ),
        call = call
      )
    }
    dt <- peek_dtype(idx)
    if (!is_dtype_int(dt) && !is_dtype_uint(dt)) {
      cli_abort(
        c(
          "Every start index must have an integer data type.",
          x = "Index {i} is {.val {as.character(dt)}}.",
          i = "Convert it with {.fn nv_convert}."
        ),
        call = call
      )
    }
  }
  invisible(NULL)
}

# Assert that `value` is a scalar, for the operands whose spec allows a single
# value only (a padding value, a reduction's `init`). stablehlo reports these
# as "0-dimensional", the word anvl's pages do not use.
assert_arrayish_scalar <- function(value, arg = rlang::caller_arg(value), call = rlang::caller_env()) {
  if (naxes(value) != 0L) {
    cli_abort(
      c(
        "{.arg {arg}} must be a scalar.",
        x = "Got shape {shape_repr(shape(value))}."
      ),
      call = call
    )
  }
  invisible(NULL)
}

# Assert that `axes_arg` names one entry per axis of `x`, for the arguments
# stablehlo checks against the operand rank and reports under its own name.
assert_per_axis <- function(x, value, arg = rlang::caller_arg(value), call = rlang::caller_env()) {
  if (length(value) != naxes(x)) {
    cli_abort(
      c(
        "{.arg {arg}} must have one entry per axis of {.arg x}.",
        x = "{.arg x} has {naxes(x)} ax{?is/es}, but {.arg {arg}} has {length(value)} entr{?y/ies}."
      ),
      call = call
    )
  }
  invisible(NULL)
}

# Assert that the named operands have the same shape, allowing a scalar among
# them where `scalar_ok` says so. The primitives do not broadcast, so a
# mismatch is an error either way -- but stablehlo's inference names *its*
# operands (`on_true`, `on_false`, `min`, `max`, `lhs`, `rhs`), which no anvl
# caller wrote. Checking here keeps the caller's own argument in the message.
assert_shapes_agree <- function(..., scalar_ok = FALSE, call = rlang::caller_env()) {
  args <- list(...)
  shapes <- lapply(args, shape)
  if (scalar_ok) {
    keep <- lengths(shapes) > 0L
    args <- args[keep]
    shapes <- shapes[keep]
  }
  if (length(shapes) < 2L || all(vapply(shapes[-1L], identical, logical(1L), shapes[[1L]]))) {
    return(invisible(NULL))
  }
  got <- paste0(
    "`",
    names(args),
    "` is ",
    vapply(shapes, shape_repr, character(1L)),
    collapse = ", "
  )
  all_of <- if (length(args) > 2L) "all " else ""
  headline <- if (scalar_ok) {
    "{.arg {names(args)}} must {all_of}have the same shape, apart from any scalar among them."
  } else {
    "{.arg {names(args)}} must {all_of}have the same shape."
  }
  cli_abort(c(headline, x = "Got {got}."), call = call)
}

# Assert that a variadic function was given at least one array. Without this an
# empty `...` reaches `max()`, `Reduce()` or stablehlo and produces a warning or
# a raw backend message.
assert_some_arrays <- function(..., call = rlang::caller_env()) {
  if (...length() == 0L) {
    cli_abort(
      "At least one array is required, but none was given.",
      call = call
    )
  }
  invisible(NULL)
}

# Assert that `axis` of `x` holds elements. Operations that read a position
# along the axis (a cumulative op, a quantile) have nothing to read otherwise,
# and the backend's complaint names its own window arguments.
assert_nonempty_axis <- function(x, axis, arg = rlang::caller_arg(x), call = rlang::caller_env()) {
  shp <- shape(x)
  if (length(shp) >= axis && shp[[axis]] == 0L) {
    cli_abort(
      c(
        "{.arg {arg}} must have elements along the axis this reads.",
        x = "{.arg {arg}} has shape {shape_repr(shp)}; axis {axis} has size 0."
      ),
      call = call
    )
  }
  invisible(x)
}

# Assert `x` has exactly two axes, and optionally that it is square. Unlike
# `assert_linalg_matrix()` this says nothing about the data type, so it serves
# the operations that work at any of them.
assert_matrix <- function(x, arg = rlang::caller_arg(x), square = FALSE) {
  shp <- shape(x)
  if (length(shp) != 2L) {
    cli_abort(c(
      "{.arg {arg}} must be a matrix with exactly 2 axes.",
      x = "Got shape {shape_repr(shp)}."
    ))
  }
  if (square && shp[1L] != shp[2L]) {
    cli_abort(c(
      "{.arg {arg}} must be a square matrix.",
      x = "Got shape {shape_repr(shp)}."
    ))
  }
  invisible(x)
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
