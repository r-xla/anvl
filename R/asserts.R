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

# Convert `x` to a DataType via `as_dtype()` and assert it is a floating-point
# dtype (f32 or f64). Returns the converted DataType.
assert_float_dtype <- function(x, arg = rlang::caller_arg(x)) {
  dt <- as_dtype(x)
  if (!inherits(dt, "FloatType")) {
    cli_abort(c(
      "{.arg {arg}} must be a floating-point dtype (f32 or f64).",
      "x" = "Got {.val {as.character(dt)}}."
    ))
  }
  dt
}

assert_linalg_matrix <- function(operand, arg, square = FALSE) {
  s <- shape(operand)
  if (length(s) != 2L) {
    cli_abort(c(
      "{.arg {arg}} must be a 2-D matrix.",
      "x" = "Got shape {xlamisc::shapevec_repr(s)}."
    ))
  }
  if (any(s == 0L)) {
    cli_abort(c(
      "{.arg {arg}} must not have any zero-sized axis.",
      "x" = "Got shape {xlamisc::shapevec_repr(s)}."
    ))
  }
  if (square && s[[1L]] != s[[2L]]) {
    cli_abort(c(
      "{.arg {arg}} must be a square matrix.",
      "x" = "Got shape {xlamisc::shapevec_repr(s)}."
    ))
  }
  if (!inherits(dtype(operand), "FloatType")) {
    cli_abort(c(
      "{.arg {arg}} must have a floating-point dtype.",
      "x" = "Got dtype {.val {as.character(dtype(operand))}}."
    ))
  }
  invisible(NULL)
}
