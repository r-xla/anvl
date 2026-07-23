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
assert_same_type <- function(lhs, rhs) {
  if (dtype(lhs) != dtype(rhs)) {
    cli_abort(c(
      "`lhs` and `rhs` must have the same dtype.",
      x = "Got {.val {as.character(dtype(lhs))}} and {.val {as.character(dtype(rhs))}}."
    ))
  }
  if (!identical(shape(lhs), shape(rhs))) {
    cli_abort(c(
      "`lhs` and `rhs` must have the same shape.",
      x = "Got {xlamisc::shapevec_repr(shape(lhs))} and {xlamisc::shapevec_repr(shape(rhs))}."
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
