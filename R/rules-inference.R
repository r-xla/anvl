# Anvl-native type inference for elementwise primitives.
#
# These helpers take `AbstractArray`(s) directly and return a single
# `AbstractArray` -- no `stablehlo::ValueType` involved, no ambiguity
# handling (that's owned by `make_binary_op()` / `make_unary_op()` in
# `primitives.R`).

# Assert `x` (an `AbstractArray`) has a dtype inheriting from one of the
# given tengen dtype classes (e.g. "FloatType", "IntegerType").
assert_dtype_class <- function(x, ..., arg = rlang::caller_arg(x)) {
  classes <- c(...)
  if (!inherits(dtype(x), classes)) {
    cli_abort(c(
      "{.arg {arg}} must have dtype {.or {classes}}.",
      x = "Got {.val {as.character(dtype(x))}}."
    ))
  }
  invisible(x)
}

# Assert two `AbstractArray`s have identical dtype and shape.
assert_same_type <- function(a, b, arg_a = "lhs", arg_b = "rhs") {
  if (dtype(a) != dtype(b)) {
    cli_abort(c(
      "{.arg {arg_a}} and {.arg {arg_b}} must have the same dtype.",
      x = "Got {.val {as.character(dtype(a))}} and {.val {as.character(dtype(b))}}."
    ))
  }
  if (!identical(shape(a), shape(b))) {
    cli_abort(c(
      "{.arg {arg_a}} and {.arg {arg_b}} must have the same shape.",
      x = "Got {xlamisc::shapevec_repr(shape(a))} and {xlamisc::shapevec_repr(shape(b))}."
    ))
  }
  invisible(NULL)
}

# Infer for elementwise binary operations with no dtype restriction.
infer_generic_biv <- function(lhs, rhs) {
  assert_same_type(lhs, rhs)
  lhs
}

# Infer for elementwise float binary operations.
infer_float_biv <- function(lhs, rhs) {
  assert_same_type(lhs, rhs)
  assert_dtype_class(lhs, "FloatType")
  lhs
}

# Infer for elementwise integerish (boolean or integer) binary operations.
infer_integerish_biv <- function(lhs, rhs) {
  assert_dtype_class(lhs, "BooleanType", "IntegerType", "UIntegerType")
  assert_dtype_class(rhs, "BooleanType", "IntegerType", "UIntegerType")
  assert_same_type(lhs, rhs)
  lhs
}

# Infer for elementwise numeric (float or integer) binary operations.
infer_numeric_biv <- function(lhs, rhs) {
  assert_dtype_class(lhs, "FloatType", "IntegerType", "UIntegerType")
  assert_same_type(lhs, rhs)
  lhs
}

# Infer for elementwise integerish (boolean or integer) unary operations.
infer_integerish_uni <- function(x) {
  assert_dtype_class(x, "BooleanType", "IntegerType", "UIntegerType")
  x
}

# Infer for elementwise float unary operations.
infer_float_uni <- function(x) {
  assert_dtype_class(x, "FloatType")
  x
}

# Infer for elementwise integer (signed or unsigned) unary operations.
infer_integer_uni <- function(x) {
  assert_dtype_class(x, "IntegerType", "UIntegerType")
  x
}

# Infer for elementwise numeric (float or integer) unary operations.
infer_numeric_uni <- function(x) {
  assert_dtype_class(x, "FloatType", "IntegerType", "UIntegerType")
  x
}

# --- Custom elementwise / shape-preserving ops -------------------------------

# abs / sign: float or (signed) integer, type preserved.
infer_abs <- function(x) {
  assert_dtype_class(x, "FloatType", "IntegerType")
  x
}
infer_sign <- function(x) {
  assert_dtype_class(x, "FloatType", "IntegerType")
  x
}

# is_finite: float input, boolean output of the same shape.
infer_is_finite <- function(x) {
  assert_dtype_class(x, "FloatType")
  AbstractArray("bool", shape(x))
}

# polygamma(n, x): float, same shape/dtype for both args.
infer_polygamma <- function(n, x) {
  infer_float_biv(n, x)
}

# clamp(min, x, max): `min`/`max` must share `x`'s dtype and either match its
# shape or be scalars. Result has `x`'s type.
infer_clamp <- function(min, x, max) {
  if (dtype(min) != dtype(x) || dtype(max) != dtype(x)) {
    cli_abort(c(
      "{.arg min}, {.arg x}, and {.arg max} must have the same dtype.",
      x = "Got {.val {as.character(dtype(min))}}, {.val {as.character(dtype(x))}}, and {.val {as.character(dtype(max))}}."
    ))
  }
  xs <- shape(x)
  if (naxes(min) != 0L && !identical(shape(min), xs)) {
    cli_abort(c(
      "{.arg min} must have the same shape as {.arg x} or be a scalar.",
      x = "Got shapes {xlamisc::shapevec_repr(shape(min))} and {xlamisc::shapevec_repr(xs)}."
    ))
  }
  if (naxes(max) != 0L && !identical(shape(max), xs)) {
    cli_abort(c(
      "{.arg max} must have the same shape as {.arg x} or be a scalar.",
      x = "Got shapes {xlamisc::shapevec_repr(shape(max))} and {xlamisc::shapevec_repr(xs)}."
    ))
  }
  x
}

# compare(lhs, rhs, comparison_direction): equal-typed inputs, boolean output.
infer_compare <- function(lhs, rhs, comparison_direction) {
  valid <- c("EQ", "NE", "GE", "GT", "LE", "LT")
  if (!(comparison_direction %in% valid)) {
    cli_abort(c(
      "{.arg comparison_direction} must be one of {.val {valid}}.",
      x = "Got {.val {comparison_direction}}."
    ))
  }
  assert_same_type(lhs, rhs)
  AbstractArray("bool", shape(lhs))
}

# select(pred, true_value, false_value): branches must share a type; `pred` is
# boolean and either a scalar or the same shape as the branches.
infer_select <- function(pred, true_value, false_value) {
  assert_same_type(true_value, false_value, "true_value", "false_value")
  assert_dtype_class(pred, "BooleanType")
  if (naxes(pred) != 0L && !identical(shape(pred), shape(true_value))) {
    cli_abort(c(
      "{.arg pred} must be a scalar or have the same shape as {.arg true_value}.",
      x = "Got shapes {xlamisc::shapevec_repr(shape(pred))} and {xlamisc::shapevec_repr(shape(true_value))}."
    ))
  }
  true_value
}

# bitcast_convert(x, dtype): reinterpret the bytes of `x` as `dtype`. When the
# target is wider, the last axis (whose size must equal the width ratio) is
# consumed; when narrower, a trailing axis is appended.
infer_bitcast_convert <- function(x, dtype) {
  dtype <- as.character(dtype)
  if (inherits(dtype(x), "BooleanType")) {
    cli_abort(c(
      "Bitcast conversions from and to i1 are not supported.",
      x = "{.arg x} has dtype {.val {as.character(dtype(x))}}."
    ))
  }
  if (dtype %in% c("i1", "pred", "bool")) {
    cli_abort(c(
      "Bitcast conversions from and to i1 are not supported.",
      x = "{.arg dtype} is {.val {as.character(as_dtype(dtype))}}."
    ))
  }
  supported <- c(
    "i8",
    "i16",
    "i32",
    "i64",
    "ui8",
    "ui16",
    "ui32",
    "ui64",
    "f8",
    "f16",
    "f32",
    "f64"
  )
  if (!(dtype %in% supported)) {
    cli_abort("Unsupported dtype: {dtype}")
  }
  bits <- function(dt) as.integer(sub(".*?([0-9]+)$", "\\1", as.character(dt)))
  cst_fct <- bits(dtype) / bits(dtype(x))
  shp <- shape(x)
  if (cst_fct == 1) {
    result_shape <- shp
  } else if (cst_fct > 1) {
    if (length(shp) == 0L) {
      cli_abort(c(
        "{.arg x} must have at least one axis for this bitcast conversion.",
        x = "{.arg x} is a scalar ({.val {as.character(dtype(x))}} -> {.val {dtype}})."
      ))
    } else if (shp[[length(shp)]] != cst_fct) {
      cli_abort(c(
        "The last axis of {.arg x} must have size {cst_fct} for this bitcast conversion.",
        x = "Got size {shp[[length(shp)]]}."
      ))
    } else {
      result_shape <- shp[seq_len(length(shp) - 1L)]
    }
  } else {
    result_shape <- c(shp, as.integer(1 / cst_fct))
  }
  AbstractArray(dtype, result_shape)
}
