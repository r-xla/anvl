REVIEW: claude-config/AGENTS.md to keep NEWS entries short and in-line with the existing style in other NEWS.md entries.
#' @include api.R

# REVIEW: I want ONE doc page per generic.
# Use templates when (parts of) documentation are duplicated.


#' @title Base R Generics for anvl Arrays
#' @name anvl-generics
#' @description
#' [`AnvlArray`]s and [`AnvlBox`]es (the traced values inside [jit()]) implement
#' most of base R's generic functions -- the arithmetic and comparison
#' operators, the mathematical functions, the reductions and the array generics
#' such as `c()`, `dim()` or `t()` -- and mean what they mean in base R.
#'
#' This page documents the ones that deliberately do something else. Everything
#' not listed here behaves like its base R counterpart, so base R's
#' documentation applies; the `nv_*` function a generic delegates to (e.g.
#' [nv_add()] for `+`, [nv_sqrt()] for `sqrt()`) describes the details.
#'
#' @section Logical operators and reductions:
#' `&`, `|`, `!`, `any()` and `all()` are *logical*, like in base R, and
#' therefore return a boolean array. Unlike base R they do not coerce a
#' non-boolean operand: an integer or float array is an error rather than a
#' comparison against zero, just like [sqrt()] does not promote an integer
#' array. Write the comparison yourself (`x != 0`) if that is what you mean.
#'
#' The named functions [nv_and()], [nv_or()], [nv_xor()] and [nv_not()] are
#' *bitwise* instead -- they operate on the bits of an integer array, and
#' coincide with the operators on a boolean array.
#'
#' ```
#' nv_array(c(TRUE, FALSE)) & nv_array(c(TRUE, TRUE))  # TRUE FALSE
#' nv_array(12L) & nv_array(10L)  # error (12L & 10L is TRUE in base R)
#' nv_and(nv_array(12L), nv_array(10L))  # 8 (bitwise)
#' ```
#'
#' `xor()` is a plain function in base R, built on `|` and `&`, and therefore
#' behaves the same way.
#'
#' @section Rounding:
#' `round()`, `signif()`, `floor()`, `ceiling()` and `trunc()` require a float
#' array. Base R leaves an integer alone, but on an anvl array, where an
#' integer is never silently promoted, rounding one is a no-op and almost
#' always a missing [nv_convert()].
#'
#' `round(x, digits)` is computed by scaling with `10^digits`, so it can differ
#' from base R's rounding in the last representable digit.
#'
#' @section Reductions:
#' `sum()`, `prod()`, `max()`, `min()` and `range()` reduce over *all* axes and,
#' like in base R, accept several data arguments: `sum(x, y)` is the sum of both
#' arrays. `na.rm` is forwarded to the `nan_rm` argument of the underlying
#' `nv_reduce_*()` function. Beyond base R, named arguments are passed on, so
#' `sum(x, axes = 1L)` reduces a single axis -- but only when `x` is the only
#' data argument.
#'
#' `cumsum()`, `cumprod()`, `cummax()` and `cummin()` flatten a multi-axis array
#' in row-major order, whereas base R uses column-major order (see the "Gotchas"
#' vignette).
#'
#' @section Gamma:
#' XLA has no gamma function, so `gamma()` is computed as `exp(lgamma())` (with
#' Euler's reflection formula for a negative argument) and is therefore less
#' accurate than base R's.
#'
#' @section Other deliberate differences:
#' * **No recycling.** Only scalars broadcast; see the "Gotchas" vignette.
#' * **No `NA`.** XLA has no missing value, so a boolean array is always `TRUE`
#'   or `FALSE` and `&`, `|`, `!`, `any()` and `all()` never see a missing
#'   value.
#' * **Data types do not follow R's coercions.** An integer array is not
#'   promoted to a float array, so `sqrt()`, `log()`, `gamma()`, ... require a
#'   float array -- convert with [nv_convert()]. Reductions that are inherently
#'   fractional ([nv_mean()], [nv_median()], [nv_quantile()]) do compute at the
#'   default float, like base R.
#' * **`median()` and `sort()` default to the last axis** rather than flattening
#'   the whole array, unlike base R.
#' * **`t()` requires a matrix**, whereas base R also transposes a vector (into
#'   a 1-row matrix) and reverses the axes of a higher-rank array.
#' * **`dim()` returns the shape**, so `nrow()` and `ncol()` report the size of
#'   axis 1 and 2 -- also for a 1-D array, where base R would return `NULL`.
#' * **`[` does not accept negative or out-of-bounds indices** and has no `drop`
#'   argument; see the "Subsetting" vignette.
#' * **`as.vector()` only supports `mode = "any"`**, and `as.double()`,
#'   `as.integer()` and `as.logical()` extract an R vector of a matching data
#'   type instead of converting -- use [nv_convert()] to change the data type.
#' @param x,e1,e2 ([`arrayish`])\cr Operands.
#' @param digits (`numeric(1)`)\cr Number of digits, as in [base::round()] and
#'   [base::signif()]. Must be a plain R value.
#' @param na.rm (`logical(1)`)\cr Forwarded to the `nan_rm` argument of the
#'   underlying `nv_reduce_*()` function.
#' @param ... For the reductions, further arrays to reduce, plus named arguments
#'   for the underlying `nv_reduce_*()` function (e.g. `axes`). For `round()`
#'   and `trunc()`, further arguments of [nv_round()] / [nv_trunc()].
#' @return [`arrayish`]
#' @seealso [nv_and()] and [nv_not()] for the bitwise operations,
#'   [nv_convert()] for data type conversion.
NULL

## Move asserts into asserts.R, keep then more generic without this over-specialized error message.

# Base R's `&`, `|` and `!` are logical operators, so anvl's are too: they
# require a boolean operand rather than coercing a numeric one, the same way
# `sqrt()` requires a float array instead of promoting an integer one. The
# named functions `nv_and()` / `nv_or()` / `nv_not()` are bitwise, and on a
# boolean array bitwise and logical coincide.
assert_boolean_operand <- function(x, arg, what) {
  if (!is_arrayish(x, convert_ok = FALSE)) {
    if (!is.logical(x)) {
      cli_abort(c(
        "{.code {what}} requires a boolean operand.",
        "x" = "{.arg {arg}} is {.obj_type_friendly {x}}.",
        "i" = "Compare it explicitly, e.g. {.code x != 0}."
      ))
    }
    return(x)
  }
  if (!is_dtype_bool(peek_dtype(x))) {
    # Only the operators have a bitwise counterpart to point at.
    bitwise <- if (what %in% c("&", "|", "!")) {
      "{.fn nv_and}, {.fn nv_or}, {.fn nv_xor} and {.fn nv_not} operate on the bits of an integer array."
    }
    cli_abort(c(
      "{.code {what}} requires a boolean array.",
      "x" = "{.arg {arg}} has data type {.val {as.character(peek_dtype(x))}}.",
      "i" = "Compare it explicitly, e.g. {.code x != 0}, or convert it with {.fn nv_convert}.",
      "i" = bitwise
    ))
  }
  x
}

# Base R leaves an integer alone here, but on an anvl array rounding to whole
# numbers would then be a no-op the user did not mean to write -- almost
# always a missing `nv_convert()` -- so ask for a float array instead.
assert_float_array <- function(x, what) {
  if (!is_dtype_float(peek_dtype(x))) {
    cli_abort(c(
      "{.fn {what}} requires a float array.",
      "x" = "Got data type {.val {as.character(peek_dtype(x))}}.",
      "i" = "Convert it with {.fn nv_convert}."
    ))
  }
  invisible(x)
}

# Arithmetic operators ---------------------------------------------------------

#

# REVIEW: Generics that are also available via a nv_ function should all share the help page via "@rdname" and @usage NULL
# Those where the R generic / API function differs from base R's implementation should document the difference.

#' @rdname nv_add
#' @usage NULL
#' @export
`+.AnvlArray` <- function(e1, e2) {
  # Base R's unary `+` is the identity.
  if (missing(e2)) e1 else nv_add(e1, e2)
}

#' @export
`+.AnvlBox` <- `+.AnvlArray`

#' @export
`-.AnvlArray` <- function(e1, e2) {
  if (missing(e2)) nv_negate(e1) else nv_sub(e1, e2)
}

#' @export
`-.AnvlBox` <- `-.AnvlArray`

#' @export
`*.AnvlArray` <- function(e1, e2) {
  nv_mul(e1, e2)
}

#' @export
`*.AnvlBox` <- `*.AnvlArray`

#' @export
`/.AnvlArray` <- function(e1, e2) {
  nv_div(e1, e2)
}

#' @export
`/.AnvlBox` <- `/.AnvlArray`

#' @export
`^.AnvlArray` <- function(e1, e2) {
  nv_pow(e1, e2)
}

#' @export
`^.AnvlBox` <- `^.AnvlArray`

#' @export
`%%.AnvlArray` <- function(e1, e2) {
  nv_mod(e1, e2)
}

#' @export
`%%.AnvlBox` <- `%%.AnvlArray`

#' @export
`%/%.AnvlArray` <- function(e1, e2) {
  # Rename to nv_floor_div
  nv_int_div(e1, e2)
}

#' @export
`%/%.AnvlBox` <- `%/%.AnvlArray`

#' @export
`%*%.AnvlArray` <- function(x, y) {
  nv_matmul(x, y)
}

#' @export
`%*%.AnvlBox` <- `%*%.AnvlArray`

# Comparison operators ---------------------------------------------------------

#' @export
`==.AnvlArray` <- function(e1, e2) {
  nv_eq(e1, e2)
}

#' @export
`==.AnvlBox` <- `==.AnvlArray`

#' @export
`!=.AnvlArray` <- function(e1, e2) {
  nv_ne(e1, e2)
}

#' @export
`!=.AnvlBox` <- `!=.AnvlArray`

#' @export
`<.AnvlArray` <- function(e1, e2) {
  nv_lt(e1, e2)
}

#' @export
`<.AnvlBox` <- `<.AnvlArray`

#' @export
`<=.AnvlArray` <- function(e1, e2) {
  nv_le(e1, e2)
}

#' @export
`<=.AnvlBox` <- `<=.AnvlArray`

#' @export
`>.AnvlArray` <- function(e1, e2) {
  nv_gt(e1, e2)
}

#' @export
`>.AnvlBox` <- `>.AnvlArray`

#' @export
`>=.AnvlArray` <- function(e1, e2) {
  nv_ge(e1, e2)
}

#' @export
`>=.AnvlBox` <- `>=.AnvlArray`

# Logical operators ------------------------------------------------------------

#' @rdname anvl-generics
#' @export
`&.AnvlArray` <- function(e1, e2) {
  nv_and(
    assert_boolean_operand(e1, "e1", "&"),
    assert_boolean_operand(e2, "e2", "&")
  )
}

#' @export
`&.AnvlBox` <- `&.AnvlArray`

#' @rdname anvl-generics
#' @export
`|.AnvlArray` <- function(e1, e2) {
  nv_or(
    assert_boolean_operand(e1, "e1", "|"),
    assert_boolean_operand(e2, "e2", "|")
  )
}

#' @export
`|.AnvlBox` <- `|.AnvlArray`

#' @rdname anvl-generics
#' @export
`!.AnvlArray` <- function(x) {
  nv_not(assert_boolean_operand(x, "x", "!"))
}

#' @export
`!.AnvlBox` <- `!.AnvlArray`

# Math generics ----------------------------------------------------------------

#' @export
abs.AnvlArray <- function(x) {
  nv_abs(x)
}

#' @export
abs.AnvlBox <- abs.AnvlArray

#' @export
sign.AnvlArray <- function(x) {
  nv_sign(x)
}

#' @export
sign.AnvlBox <- sign.AnvlArray

#' @export
sqrt.AnvlArray <- function(x) {
  nv_sqrt(x)
}

#' @export
sqrt.AnvlBox <- sqrt.AnvlArray

#' @export
exp.AnvlArray <- function(x) {
  nv_exp(x)
}

#' @export
exp.AnvlBox <- exp.AnvlArray

#' @export
expm1.AnvlArray <- function(x) {
  nv_expm1(x)
}

#' @export
expm1.AnvlBox <- expm1.AnvlArray

#' @export
log.AnvlArray <- function(x, base = exp(1)) {
  if (missing(base)) {
    return(nv_log(x))
  }
  if (is_arrayish(base, convert_ok = FALSE)) {
    return(nv_log(x) / nv_log(base))
  }
  checkmate::assert_number(base, lower = 0)
  nv_log(x) / log(base)
}

#' @export
log.AnvlBox <- log.AnvlArray

#' @export
log2.AnvlArray <- function(x) {
  nv_log2(x)
}

#' @export
log2.AnvlBox <- log2.AnvlArray

#' @export
log10.AnvlArray <- function(x) {
  nv_log10(x)
}

#' @export
log10.AnvlBox <- log10.AnvlArray

#' @export
log1p.AnvlArray <- function(x) {
  nv_log1p(x)
}

#' @export
log1p.AnvlBox <- log1p.AnvlArray

#' @export
cos.AnvlArray <- function(x) {
  nv_cos(x)
}

#' @export
cos.AnvlBox <- cos.AnvlArray

#' @export
sin.AnvlArray <- function(x) {
  nv_sin(x)
}

#' @export
sin.AnvlBox <- sin.AnvlArray

#' @export
tan.AnvlArray <- function(x) {
  nv_tan(x)
}

#' @export
tan.AnvlBox <- tan.AnvlArray

#' @export
acos.AnvlArray <- function(x) {
  nv_acos(x)
}

#' @export
acos.AnvlBox <- acos.AnvlArray

#' @export
asin.AnvlArray <- function(x) {
  nv_asin(x)
}

#' @export
asin.AnvlBox <- asin.AnvlArray

#' @export
atan.AnvlArray <- function(x) {
  nv_atan(x)
}

#' @export
atan.AnvlBox <- atan.AnvlArray

#' @export
cosh.AnvlArray <- function(x) {
  nv_cosh(x)
}

#' @export
cosh.AnvlBox <- cosh.AnvlArray

#' @export
sinh.AnvlArray <- function(x) {
  nv_sinh(x)
}

#' @export
sinh.AnvlBox <- sinh.AnvlArray

#' @export
tanh.AnvlArray <- function(x) {
  nv_tanh(x)
}

#' @export
tanh.AnvlBox <- tanh.AnvlArray

#' @export
acosh.AnvlArray <- function(x) {
  nv_acosh(x)
}

#' @export
acosh.AnvlBox <- acosh.AnvlArray

#' @export
asinh.AnvlArray <- function(x) {
  nv_asinh(x)
}

#' @export
asinh.AnvlBox <- asinh.AnvlArray

#' @export
atanh.AnvlArray <- function(x) {
  nv_atanh(x)
}

#' @export
atanh.AnvlBox <- atanh.AnvlArray

# `sinpi()` / `cospi()` are exact for (half-)integer arguments in base R, so
# reduce the argument to [-0.5, 0.5] around the nearest integer first and take
# the sign from that integer's parity.
#' @export
sinpi.AnvlArray <- function(x) {
  # REVIEW:
  # - Jit this.
  # - Also add to AGENTS.md that functions with multiple ops should be jitted
  # - Ensure that this works with ints (will happen when nv_round accepts ints)
  x <- as_anvl_array(x)
  n <- nv_round(x, method = "nearest_even")
  reduced <- nv_sin((x - n) * pi)
  nv_ifelse(nv_mod(n, 2) == 0, reduced, -reduced)
}

#' @export
sinpi.AnvlBox <- sinpi.AnvlArray

#' @export
#'
# REVIEW: Jit
cospi.AnvlArray <- function(x) {
  # cos(pi * x) == sin(pi * (x + 1/2))
  sinpi(as_anvl_array(x) + 0.5)
}

#' @export
cospi.AnvlBox <- cospi.AnvlArray

#' @export
tanpi.AnvlArray <- function(x) {
  # REVIEW: Jit
  x <- as_anvl_array(x)
  denominator <- cospi(x)
  # Base R returns NaN at the half-integers, where the tangent has its poles.
  nv_ifelse(denominator == 0, NaN, sinpi(x) / denominator)
}

#' @export
tanpi.AnvlBox <- tanpi.AnvlArray

#' @export
lgamma.AnvlArray <- function(x) {
  nv_lgamma(x)
}

#' @export
lgamma.AnvlBox <- lgamma.AnvlArray

#' @export
digamma.AnvlArray <- function(x) {
  nv_digamma(x)
}

#' @export
digamma.AnvlBox <- digamma.AnvlArray

#' @export
trigamma.AnvlArray <- function(x) {
  nv_polygamma(1, x)
}

#' @export
trigamma.AnvlBox <- trigamma.AnvlArray

#' @rdname anvl-generics
#' @export
gamma.AnvlArray <- function(x) {
  # REVIEW: promote ints to float
  # REVIEW: Is this formula really correct? ensure good tests in test-api-generics.R
  x <- as_anvl_array(x)
  # XLA has no gamma, only lgamma. For a negative argument, where lgamma is the
  # log of the *absolute* value, use Euler's reflection formula
  # gamma(x) * gamma(1 - x) = pi / sin(pi * x) instead.
  positive <- nv_exp(nv_lgamma(x))
  reflected <- pi / (sinpi(x) * nv_exp(nv_lgamma(1 - x)))
  out <- nv_ifelse(x < 0, reflected, positive)
  # gamma has a pole at every non-positive integer; base R returns NaN there.
  nv_ifelse((x <= 0) & (x == nv_floor(x)), NaN, out)
}

#' @export
gamma.AnvlBox <- gamma.AnvlArray

#' @rdname anvl-generics
#' @export
floor.AnvlArray <- function(x) {
  # REVIEW: Actually, let ints through as well
  assert_float_array(x, "floor")
  nv_floor(x)
}

#' @export
floor.AnvlBox <- floor.AnvlArray

#' @rdname anvl-generics
#' @export
ceiling.AnvlArray <- function(x) {
  # REVIEW: Actually, let ints through as well
  assert_float_array(x, "ceiling")
  nv_ceiling(x)
}

#' @export
ceiling.AnvlBox <- ceiling.AnvlArray

#' @rdname anvl-generics
#' @export
trunc.AnvlArray <- function(x, ...) {
  # REVIEW: Actually, let ints through as well
  assert_float_array(x, "trunc")
  nv_trunc(x, ...)
}

#' @export
trunc.AnvlBox <- trunc.AnvlArray

#' @rdname anvl-generics
#' @export
round.AnvlArray <- function(x, digits = 0, ...) {
  # REVIEW: Actually, let ints through as well
  checkmate::assert_number(digits, finite = TRUE)
  assert_float_array(x, "round")
  if (digits == 0) {
    return(nv_round(x, ...))
  }
  scale <- 10^digits
  nv_round(x * scale, ...) / scale
}

#' @export
round.AnvlBox <- round.AnvlArray

#' @rdname anvl-generics
#' @export
signif.AnvlArray <- function(x, digits = 6) {
  checkmate::assert_number(digits, finite = TRUE)
  # Like base R, which warns and uses 1 for a smaller value.
  digits <- max(digits, 1)
  assert_float_array(x, "signif")
  x <- as_anvl_array(x)
  # Round the mantissa: shift the value so that `digits` significant digits
  # sit in front of the decimal point, round there, and shift back.
  scale <- nv_pow(10, digits - 1 - nv_floor(nv_log10(nv_abs(x))))
  rounded <- nv_round(x * scale, method = "nearest_even") / scale
  # 0 has no magnitude, and Inf / NaN must pass through unchanged.
  nv_ifelse(nv_is_finite(x) & (x != 0), rounded, x)
}

#' @export
signif.AnvlBox <- signif.AnvlArray

#' @rdname anvl-generics
#' @export
cumsum.AnvlArray <- function(x) {
  nv_cumsum(x)
}

#' @export
cumsum.AnvlBox <- cumsum.AnvlArray

#' @rdname anvl-generics
#' @export
cumprod.AnvlArray <- function(x) {
  nv_cumprod(x)
}

#' @export
cumprod.AnvlBox <- cumprod.AnvlArray

#' @rdname anvl-generics
#' @export
cummax.AnvlArray <- function(x) {
  nv_cummax(x)
}

#' @export
cummax.AnvlBox <- cummax.AnvlArray

#' @rdname anvl-generics
#' @export
cummin.AnvlArray <- function(x) {
  nv_cummin(x)
}

#' @export
cummin.AnvlBox <- cummin.AnvlArray

# Summary generics -------------------------------------------------------------

# Like base R, every unnamed argument is data: `sum(x, y)` sums both. Named
# arguments are options of the underlying nv_reduce_* (e.g. `axes = 1L`), and
# `na.rm` becomes its `nan_rm`; unsupported ones error there as unused args.
# REVIEW: Not all the logic in one function.
summary_generic <- function(op, args, na.rm) {
  named <- nzchar(names(args) %||% rep("", length(args)))
  data <- args[!named]
  opts <- args[named]
  if (length(data) > 1L && length(opts) > 0L) {
    cli_abort(c(
      "{.fn {op}} cannot combine several data arguments with {.arg {names(opts)[1L]}}.",
      "i" = "Reduce the arrays one at a time, e.g. {.code {op}(x, axes = 1L)}."
    ))
  }
  # REVIEW: THis logic should be in range.AnvlArray, not here.
  if (op == "range") {
    return(nv_concatenate(
      summary_reduce("min", data, opts, na.rm),
      summary_reduce("max", data, opts, na.rm)
    ))
  }
  summary_reduce(op, data, opts, na.rm)
}

# Reduce each data argument over all its axes, then combine the results
# element-wise, the way base R combines its arguments.

# REVIEW: THis is a weird implementation. Instead use a helper function that can be re-used
# in the different summary generics, but don't hard-code all options into one common helper
# the op-specific logic should be in the specific generic impl
summary_reduce <- function(op, data, opts, na.rm) {
  what <- paste0(op, "()")
  parts <- lapply(data, function(z) {
    if (!is_arrayish(z, convert_ok = FALSE)) {
      # A plain R value is reduced in R, and enters as a literal afterwards.
      return(switch(
        op,
        "max" = max(z, na.rm = na.rm),
        "min" = min(z, na.rm = na.rm),
        "prod" = prod(z, na.rm = na.rm),
        "sum" = sum(z, na.rm = na.rm),
        # `any()` / `all()` are logical, so a non-logical value is rejected
        # rather than coerced.
        "any" = any(assert_boolean_operand(z, "x", what), na.rm = na.rm),
        "all" = all(assert_boolean_operand(z, "x", what), na.rm = na.rm)
      ))
    }
    switch(
      op,
      "max" = do.call(nv_reduce_max, c(list(z), opts, list(nan_rm = na.rm))),
      "min" = do.call(nv_reduce_min, c(list(z), opts, list(nan_rm = na.rm))),
      "prod" = do.call(nv_reduce_prod, c(list(z), opts, list(nan_rm = na.rm))),
      "sum" = do.call(nv_reduce_sum, c(list(z), opts, list(nan_rm = na.rm))),
      # `any()` / `all()` are logical, so they require a boolean array and
      # have no `nan_rm`.
      "any" = do.call(nv_reduce_any, c(list(assert_boolean_operand(z, "x", what)), opts)),
      "all" = do.call(nv_reduce_all, c(list(assert_boolean_operand(z, "x", what)), opts))
    )
  })
  combine <- switch(
    op,
    "max" = nv_max,
    "min" = nv_min,
    "prod" = nv_mul,
    "sum" = nv_add,
    "any" = nv_or,
    "all" = nv_and
  )
  Reduce(combine, parts)
}

#' @rdname anvl-generics
#' @export
sum.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic("sum", list(...), na.rm)
}

#' @export
sum.AnvlBox <- sum.AnvlArray

#' @rdname anvl-generics
#' @export
prod.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic("prod", list(...), na.rm)
}

#' @export
prod.AnvlBox <- prod.AnvlArray

#' @rdname anvl-generics
#' @export
max.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic("max", list(...), na.rm)
}

#' @export
max.AnvlBox <- max.AnvlArray

#' @rdname anvl-generics
#' @export
min.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic("min", list(...), na.rm)
}

#' @export
min.AnvlBox <- min.AnvlArray

#' @rdname anvl-generics
#' @export
range.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic("range", list(...), na.rm)
}

#' @export
range.AnvlBox <- range.AnvlArray

#' @rdname anvl-generics
#' @export
any.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic("any", list(...), na.rm)
}

#' @export
any.AnvlBox <- any.AnvlArray

#' @rdname anvl-generics
#' @export
all.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic("all", list(...), na.rm)
}

#' @export
all.AnvlBox <- all.AnvlArray

# Other generics ---------------------------------------------------------------

#' @rdname nv_mean
#' @param trim Currently not supported.
#' @param na.rm Forwarded to [nv_mean()]'s `nan_rm` argument.
#' @param ... No additional arguments.
#' @method mean AnvlArray
#' @export
mean.AnvlArray <- function(x, trim = 0, na.rm = FALSE, ..., axes = NULL, drop = TRUE) {
  if (!identical(trim, 0)) {
    cli_abort("{.arg trim} is not supported by {.fn mean} for anvl arrays.")
  }
  nv_mean(x, ..., axes = axes, drop = drop, nan_rm = na.rm)
}

#' @method mean AnvlBox
#' @export
mean.AnvlBox <- mean.AnvlArray

#' @rdname nv_is_nan
#' @method is.nan AnvlArray
#' @export
is.nan.AnvlArray <- function(x) {
  nv_is_nan(x)
}

#' @method is.nan AnvlBox
#' @export
is.nan.AnvlBox <- is.nan.AnvlArray

#' @rdname nv_is_infinite
#' @method is.infinite AnvlArray
#' @export
is.infinite.AnvlArray <- function(x) {
  nv_is_infinite(x)
}

#' @method is.infinite AnvlBox
#' @export
is.infinite.AnvlBox <- is.infinite.AnvlArray

#' @rdname nv_is_finite
#' @method is.finite AnvlArray
#' @export
is.finite.AnvlArray <- function(x) {
  nv_is_finite(x)
}

#' @method is.finite AnvlBox
#' @export
is.finite.AnvlBox <- is.finite.AnvlArray

# if we don't give it the name nv_transpose, pkgdown thinks t.anvl is a package

#' @title Transpose
#' @name nv_transpose
#' @description
#' Permutes the axes of an array. You can also use `t()` for matrices.
#' @param permutation (`integer()` | `NULL`)\cr
#'   New ordering of axes. If `NULL` (default), reverses the axes.
#'   Negative values count from the end, i.e. `-1` refers to the last axis.
#' @return [`arrayish`]\cr
#'   Has the same data type as `x` and shape `nv_shape(x)[permutation]`.
#' @seealso [prim_transpose()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' t(x)
#' @method t AnvlArray
#' @export
t.AnvlArray <- function(x) {
  nd <- naxes(x)
  if (nd != 2L) {
    cli_abort("{.fn t} requires a 2-D array, but got a {nd}-D array.")
  }
  nv_transpose(x)
}

#' @method t AnvlBox
#' @export
t.AnvlBox <- t.AnvlArray

#' @rdname nv_reverse
#' @section Relation to base R:
#' `rev()` reverses along every axis, which puts the elements in the same
#' order as [base::rev()] does (base R flattens the array to a vector first,
#' whereas `rev()` on an anvl array keeps the shape).
#' @method rev AnvlArray
#' @export
rev.AnvlArray <- function(x) {
  if (naxes(x) == 0L) {
    # A scalar array has nothing to reverse.
    return(x)
  }
  nv_reverse(x, axes = seq_len(naxes(x)))
}

#' @method rev AnvlBox
#' @export
rev.AnvlBox <- rev.AnvlArray

#' @rdname nv_concatenate
#' @section Relation to base R:
#' `c()` flattens every argument to 1-D and concatenates the results, like
#' [base::c()] does. An anvl array is flattened in row-major order, so for an
#' input with more than one axis the elements come out in a different order
#' than base R's column-major flatten would produce (see the "Gotchas"
#' vignette). Use `nv_concatenate()` with an explicit `axis` to keep the axes.
#' @method c AnvlArray
#' @export
c.AnvlArray <- function(...) {
  # REVIEW:
  # - DO NOT FLATTEN ANYTHING. Only accept scalars and 1d vectors.
  # - Also, jit the function
  args <- lapply(list(...), function(x) {
    if (is_arrayish(x, convert_ok = FALSE)) {
      if (naxes(x) > 1L) nv_flatten(x) else x
    } else if (is.atomic(x) && length(x) != 1L) {
      # Base R flattens an R array in column-major order, and a plain vector
      # has to become a 1-D R array to be arrayish at all.
      array(as.vector(x))
    } else {
      x
    }
  })
  do.call(nv_concatenate, c(args, list(axis = 1L)))
}

#' @method c AnvlBox
#' @export
c.AnvlBox <- c.AnvlArray

#' @rdname nv_median
#' @param na.rm Forwarded to [nv_median()]'s `nan_rm` argument.
#' @param ... No additional arguments.
#' @method median AnvlArray
#' @export
median.AnvlArray <- function(x, na.rm = FALSE, ..., axis = NULL, interpolation = "linear") {
  rlang::check_dots_empty()
  nv_median(x, axis = axis, interpolation = interpolation, nan_rm = na.rm)
}

#' @method median AnvlBox
#' @export
median.AnvlBox <- median.AnvlArray

#' @rdname nv_sort
#' @param decreasing (`logical(1)`)\cr If `TRUE`, sort in decreasing order.
#' @param ... No additional arguments.
#' @method sort AnvlArray
#' @export
sort.AnvlArray <- function(x, decreasing = FALSE, ..., axis = NULL) {
  nv_sort(x, decreasing = decreasing, ..., axis = axis)
}

#' @method sort AnvlBox
#' @export
sort.AnvlBox <- sort.AnvlArray

#' @rdname nv_subset
#' @method [ AnvlArray
#' @export
`[.AnvlArray` <- function(x, ...) {
  # nargs() sees trailing missing args (e.g. the last `,` in x[1:5, , ])
  # that rlang::enquos() silently drops.
  n_args <- nargs() - 1L
  rank <- naxes(x)
  quos <- rlang::enquos(...)
  if ("drop" %in% names(quos)) {
    cli_abort(c(
      "{.arg drop} is not supported when subsetting an array.",
      "i" = "An axis selected with a scalar index is always dropped; use \
             {.fn nv_unsqueeze} to add it back."
    ))
  }
  if (n_args > rank) {
    cli_abort("Too many subset specifications: got {n_args}, expected at most {rank}")
  }
  rlang::inject(nv_subset(x, !!!quos))
}

#' @method [ AnvlBox
#' @export
`[.AnvlBox` <- `[.AnvlArray`

#' @rdname nv_subset_assign
#' @method [<- AnvlArray
#' @export
`[<-.AnvlArray` <- function(x, ..., value) {
  n_args <- nargs() - 2L
  rank <- naxes(x)
  if (n_args > rank) {
    cli_abort("Too many subset specifications: got {n_args}, expected at most {rank}")
  }
  quos <- rlang::enquos(...)
  rlang::inject(nv_subset_assign(x, !!!quos, value = value))
}

#' @method [<- AnvlBox
#' @export
`[<-.AnvlBox` <- `[<-.AnvlArray`

# `crossprod()`/`tcrossprod()` only became S3 generic in R 4.4.0 (`%*%` got
# there in 4.3.0), so these methods are what sets the package's minimum R
# version: on 4.3 they register but never dispatch, and `crossprod(x)` on an
# `AnvlArray` errors from base instead.
#' @rdname nv_crossprod
#' @param x,y Same as `lhs` and `rhs`; the names used by the base R S3 generic.
#' @param ... No additional arguments.
#' @method crossprod AnvlArray
#' @export
crossprod.AnvlArray <- function(x, y = NULL, ...) {
  nv_crossprod(x, y, ...)
}

#' @method crossprod AnvlBox
#' @export
crossprod.AnvlBox <- crossprod.AnvlArray

#' @rdname nv_tcrossprod
#' @param x,y Same as `lhs` and `rhs`; the names used by the base R S3 generic.
#' @param ... No additional arguments.
#' @method tcrossprod AnvlArray
#' @export
tcrossprod.AnvlArray <- function(x, y = NULL, ...) {
  nv_tcrossprod(x, y, ...)
}

#' @method tcrossprod AnvlBox
#' @export
tcrossprod.AnvlBox <- tcrossprod.AnvlArray

#' @method dim AnvlArray
#' @export
dim.AnvlArray <- function(x) {
  shape(x)
}

#' @method dim AnvlBox
#' @export
dim.AnvlBox <- dim.AnvlArray

#' @method length AnvlArray
#' @export
length.AnvlArray <- function(x) {
  prod(shape(x))
}

#' @method length AnvlBox
#' @export
length.AnvlBox <- length.AnvlArray

#' @rdname nv_bind
#' @param deparse.level Ignored. Kept for compatibility with [base::rbind()]
#'   and [base::cbind()].
#' @method rbind AnvlArray
#' @export
rbind.AnvlArray <- function(..., deparse.level = 1) {
  nv_rbind(...)
}

#' @method rbind AnvlBox
#' @export
rbind.AnvlBox <- rbind.AnvlArray

#' @rdname nv_bind
#' @method cbind AnvlArray
#' @export
cbind.AnvlArray <- function(..., deparse.level = 1) {
  nv_cbind(...)
}

#' @method cbind AnvlBox
#' @export
cbind.AnvlBox <- cbind.AnvlArray

#' @rdname nv_solve
#' @param a ([`arrayish`])\cr Coefficient matrix.
#' @param b ([`arrayish`])\cr Right-hand side. If missing, returns [nv_inv()] of `a`.
#' @param ... No additional arguments.
#' @method solve AnvlArray
#' @export
solve.AnvlArray <- function(a, b, ...) {
  if (missing(b)) nv_inv(a, ...) else nv_solve(a, b, ...)
}

#' @method solve AnvlBox
#' @export
solve.AnvlBox <- solve.AnvlArray

#' @rdname nv_qr
#' @param ... No additional arguments.
#' @method qr AnvlArray
#' @export
qr.AnvlArray <- function(x, ...) {
  nv_qr(x, ...)
}

#' @method qr AnvlBox
#' @export
qr.AnvlBox <- qr.AnvlArray

#' @rdname nv_chol
#' @param lower (`logical(1)`)\cr If `TRUE`, return the lower-triangular factor.
#' @param ... No additional arguments.
#' @method chol AnvlArray
#' @export
chol.AnvlArray <- function(x, ..., lower = FALSE) {
  nv_chol(x, lower = lower, ...)
}

#' @method chol AnvlBox
#' @export
chol.AnvlBox <- chol.AnvlArray

#' @rdname nv_determinant
#' @param logarithm (`logical(1)`)\cr If `TRUE` (default), return the log
#'   of the absolute determinant.
#' @param ... No additional arguments.
#' @method determinant AnvlArray
#' @export
determinant.AnvlArray <- function(x, logarithm = TRUE, ...) {
  nv_determinant(x, logarithm = logarithm, ...)
}

#' @method determinant AnvlBox
#' @export
determinant.AnvlBox <- determinant.AnvlArray
