#' @include api.R
NULL

# Base R's generics on an `AnvlArray` mean what they mean in base R, so every
# method here is a thin delegate to the `nv_*` function that does the work and
# is documented on its help page. A method only carries its own documentation
# where there is no such twin (`gamma()`, `signif()`, the `*pi()` trigonometry,
# `range()`, `dim()` and `length()`).
#
# The methods for `AnvlBox` -- the traced values inside `jit()` -- are the same
# functions, registered for the second class.

# Arithmetic operators ---------------------------------------------------------

#' @rdname nv_add
#' @usage NULL
#' @export
`+.AnvlArray` <- function(e1, e2) {
  # Base R's unary `+` is the identity.
  if (missing(e2)) e1 else nv_add(e1, e2)
}

#' @export
`+.AnvlBox` <- `+.AnvlArray`

#' @rdname nv_sub
#' @usage NULL
#' @section Relation to base R:
#' The unary `-x` is [nv_negate()].
#' @export
`-.AnvlArray` <- function(e1, e2) {
  if (missing(e2)) nv_negate(e1) else nv_sub(e1, e2)
}

#' @export
`-.AnvlBox` <- `-.AnvlArray`

#' @rdname nv_mul
#' @usage NULL
#' @export
`*.AnvlArray` <- function(e1, e2) {
  nv_mul(e1, e2)
}

#' @export
`*.AnvlBox` <- `*.AnvlArray`

#' @rdname nv_div
#' @usage NULL
#' @export
`/.AnvlArray` <- function(e1, e2) {
  nv_div(e1, e2)
}

#' @export
`/.AnvlBox` <- `/.AnvlArray`

#' @rdname nv_pow
#' @usage NULL
#' @export
`^.AnvlArray` <- function(e1, e2) {
  nv_pow(e1, e2)
}

#' @export
`^.AnvlBox` <- `^.AnvlArray`

#' @rdname nv_mod
#' @usage NULL
#' @export
`%%.AnvlArray` <- function(e1, e2) {
  nv_mod(e1, e2)
}

#' @export
`%%.AnvlBox` <- `%%.AnvlArray`

#' @rdname nv_floor_div
#' @usage NULL
#' @export
`%/%.AnvlArray` <- function(e1, e2) {
  nv_floor_div(e1, e2)
}

#' @export
`%/%.AnvlBox` <- `%/%.AnvlArray`

#' @rdname nv_matmul
#' @usage NULL
#' @export
`%*%.AnvlArray` <- function(x, y) {
  nv_matmul(x, y)
}

#' @export
`%*%.AnvlBox` <- `%*%.AnvlArray`

# Comparison operators ---------------------------------------------------------

#' @rdname nv_eq
#' @usage NULL
#' @export
`==.AnvlArray` <- function(e1, e2) {
  nv_eq(e1, e2)
}

#' @export
`==.AnvlBox` <- `==.AnvlArray`

#' @rdname nv_ne
#' @usage NULL
#' @export
`!=.AnvlArray` <- function(e1, e2) {
  nv_ne(e1, e2)
}

#' @export
`!=.AnvlBox` <- `!=.AnvlArray`

#' @rdname nv_lt
#' @usage NULL
#' @export
`<.AnvlArray` <- function(e1, e2) {
  nv_lt(e1, e2)
}

#' @export
`<.AnvlBox` <- `<.AnvlArray`

#' @rdname nv_le
#' @usage NULL
#' @export
`<=.AnvlArray` <- function(e1, e2) {
  nv_le(e1, e2)
}

#' @export
`<=.AnvlBox` <- `<=.AnvlArray`

#' @rdname nv_gt
#' @usage NULL
#' @export
`>.AnvlArray` <- function(e1, e2) {
  nv_gt(e1, e2)
}

#' @export
`>.AnvlBox` <- `>.AnvlArray`

#' @rdname nv_ge
#' @usage NULL
#' @export
`>=.AnvlArray` <- function(e1, e2) {
  nv_ge(e1, e2)
}

#' @export
`>=.AnvlBox` <- `>=.AnvlArray`

# Logical operators ------------------------------------------------------------

# Base R's `&`, `|` and `!` are logical operators, so anvl's are too: they
# require a boolean operand rather than coercing a numeric one, the same way
# `sqrt()` requires a float array instead of promoting an integer one. On a
# boolean array bitwise and logical coincide, so they delegate to the bitwise
# `nv_and()` / `nv_or()` / `nv_not()`.
bitwise_hint <- function() {
  c(
    "Compare it explicitly, e.g. {.code x != 0}, or convert it with {.fn nv_convert}.",
    "{.fn nv_and}, {.fn nv_or}, {.fn nv_xor} and {.fn nv_not} operate on the bits of an integer array."
  )
}

#' @rdname nv_and
#' @usage NULL
#' @section The `&` operator:
#' `&` is *logical*, like in base R: it requires boolean operands and returns a
#' boolean array, which is what `nv_and()` computes for them. Unlike base R it
#' does not coerce a non-boolean operand by comparing it against zero -- write
#' `x != 0` yourself -- because {anvl} does not apply R's data type coercions
#' anywhere else either.
#' @export
`&.AnvlArray` <- function(e1, e2) {
  nv_and(
    assert_boolean_array(e1, hint = bitwise_hint()),
    assert_boolean_array(e2, hint = bitwise_hint())
  )
}

#' @export
`&.AnvlBox` <- `&.AnvlArray`

#' @rdname nv_or
#' @usage NULL
#' @section The `|` operator:
#' `|` is *logical*, like in base R: it requires boolean operands and returns a
#' boolean array, which is what `nv_or()` computes for them. Unlike base R it
#' does not coerce a non-boolean operand by comparing it against zero -- write
#' `x != 0` yourself -- because {anvl} does not apply R's data type coercions
#' anywhere else either.
#'
#' `xor()` is a plain function in base R, built on `|` and `&`, and therefore
#' behaves the same way.
#' @export
`|.AnvlArray` <- function(e1, e2) {
  nv_or(
    assert_boolean_array(e1, hint = bitwise_hint()),
    assert_boolean_array(e2, hint = bitwise_hint())
  )
}

#' @export
`|.AnvlBox` <- `|.AnvlArray`

#' @rdname nv_not
#' @usage NULL
#' @section The `!` operator:
#' `!` is *logical*, like in base R: it requires a boolean array and returns a
#' boolean array, which is what `nv_not()` computes for one. Unlike base R it
#' does not coerce a non-boolean operand by comparing it against zero -- write
#' `x == 0` yourself -- because {anvl} does not apply R's data type coercions
#' anywhere else either.
#' @export
`!.AnvlArray` <- function(x) {
  nv_not(assert_boolean_array(x, hint = bitwise_hint()))
}

#' @export
`!.AnvlBox` <- `!.AnvlArray`

# Math generics ----------------------------------------------------------------

#' @rdname nv_abs
#' @usage NULL
#' @export
abs.AnvlArray <- function(x) {
  nv_abs(x)
}

#' @export
abs.AnvlBox <- abs.AnvlArray

#' @rdname nv_sign
#' @usage NULL
#' @export
sign.AnvlArray <- function(x) {
  nv_sign(x)
}

#' @export
sign.AnvlBox <- sign.AnvlArray

#' @rdname nv_sqrt
#' @usage NULL
#' @export
sqrt.AnvlArray <- function(x) {
  nv_sqrt(x)
}

#' @export
sqrt.AnvlBox <- sqrt.AnvlArray

#' @rdname nv_exp
#' @usage NULL
#' @export
exp.AnvlArray <- function(x) {
  nv_exp(x)
}

#' @export
exp.AnvlBox <- exp.AnvlArray

#' @rdname nv_expm1
#' @usage NULL
#' @export
expm1.AnvlArray <- function(x) {
  nv_expm1(x)
}

#' @export
expm1.AnvlBox <- expm1.AnvlArray

#' @rdname nv_log
#' @usage NULL
#' @section Relation to base R:
#' `log(x, base)` takes a second argument like [base::log()] does and computes
#' `log(x) / log(base)`.
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

#' @rdname nv_log2
#' @usage NULL
#' @export
log2.AnvlArray <- function(x) {
  nv_log2(x)
}

#' @export
log2.AnvlBox <- log2.AnvlArray

#' @rdname nv_log10
#' @usage NULL
#' @export
log10.AnvlArray <- function(x) {
  nv_log10(x)
}

#' @export
log10.AnvlBox <- log10.AnvlArray

#' @rdname nv_log1p
#' @usage NULL
#' @export
log1p.AnvlArray <- function(x) {
  nv_log1p(x)
}

#' @export
log1p.AnvlBox <- log1p.AnvlArray

#' @rdname nv_cos
#' @usage NULL
#' @export
cos.AnvlArray <- function(x) {
  nv_cos(x)
}

#' @export
cos.AnvlBox <- cos.AnvlArray

#' @rdname nv_sin
#' @usage NULL
#' @export
sin.AnvlArray <- function(x) {
  nv_sin(x)
}

#' @export
sin.AnvlBox <- sin.AnvlArray

#' @rdname nv_tan
#' @usage NULL
#' @export
tan.AnvlArray <- function(x) {
  nv_tan(x)
}

#' @export
tan.AnvlBox <- tan.AnvlArray

#' @rdname nv_acos
#' @usage NULL
#' @export
acos.AnvlArray <- function(x) {
  nv_acos(x)
}

#' @export
acos.AnvlBox <- acos.AnvlArray

#' @rdname nv_asin
#' @usage NULL
#' @export
asin.AnvlArray <- function(x) {
  nv_asin(x)
}

#' @export
asin.AnvlBox <- asin.AnvlArray

#' @rdname nv_atan
#' @usage NULL
#' @export
atan.AnvlArray <- function(x) {
  nv_atan(x)
}

#' @export
atan.AnvlBox <- atan.AnvlArray

#' @rdname nv_cosh
#' @usage NULL
#' @export
cosh.AnvlArray <- function(x) {
  nv_cosh(x)
}

#' @export
cosh.AnvlBox <- cosh.AnvlArray

#' @rdname nv_sinh
#' @usage NULL
#' @export
sinh.AnvlArray <- function(x) {
  nv_sinh(x)
}

#' @export
sinh.AnvlBox <- sinh.AnvlArray

#' @rdname nv_tanh
#' @usage NULL
#' @export
tanh.AnvlArray <- function(x) {
  nv_tanh(x)
}

#' @export
tanh.AnvlBox <- tanh.AnvlArray

#' @rdname nv_acosh
#' @usage NULL
#' @export
acosh.AnvlArray <- function(x) {
  nv_acosh(x)
}

#' @export
acosh.AnvlBox <- acosh.AnvlArray

#' @rdname nv_asinh
#' @usage NULL
#' @export
asinh.AnvlArray <- function(x) {
  nv_asinh(x)
}

#' @export
asinh.AnvlBox <- asinh.AnvlArray

#' @rdname nv_atanh
#' @usage NULL
#' @export
atanh.AnvlArray <- function(x) {
  nv_atanh(x)
}

#' @export
atanh.AnvlBox <- atanh.AnvlArray

#' @title Sine of a Multiple of Pi
#' @description
#' Element-wise `sin(pi * x)`, the generic [base::sinpi()] on an anvl array.
#' Like base R's, it is exact for a whole or half-integer argument: the
#' argument is first reduced to the interval `[-0.5, 0.5]` around the nearest
#' whole number, which is where the sine of a multiple of pi is accurate.
#' @template param_x_float
#' @template return_unary_float
#' @seealso [cospi()][cospi.AnvlArray], [tanpi()][tanpi.AnvlArray], [nv_sin()]
#' @examplesIf pjrt::plugins_downloaded()
#' sinpi(nv_array(c(0, 0.5, 1, 1.5)))
#' @method sinpi AnvlArray
#' @export
#' @jit
sinpi.AnvlArray <- function(x) {
  x <- as_anvl_array(promote_to_float(x))
  n <- nv_round(x, method = "nearest_even")
  reduced <- nv_sin((x - n) * pi)
  # The sine of `pi * n` alternates in sign with the parity of `n`.
  nv_ifelse(nv_mod(n, 2) == 0, reduced, -reduced)
}

#' @method sinpi AnvlBox
#' @export
sinpi.AnvlBox <- sinpi.AnvlArray

#' @title Cosine of a Multiple of Pi
#' @description
#' Element-wise `cos(pi * x)`, the generic [base::cospi()] on an anvl array.
#' Like base R's, it is exact for a whole or half-integer argument.
#' @template param_x_float
#' @template return_unary_float
#' @seealso [sinpi()][sinpi.AnvlArray], [tanpi()][tanpi.AnvlArray], [nv_cos()]
#' @examplesIf pjrt::plugins_downloaded()
#' cospi(nv_array(c(0, 0.5, 1, 1.5)))
#' @method cospi AnvlArray
#' @export
#' @jit
cospi.AnvlArray <- function(x) {
  # cos(pi * x) == sin(pi * (x + 1/2))
  sinpi(as_anvl_array(promote_to_float(x)) + 0.5)
}

#' @method cospi AnvlBox
#' @export
cospi.AnvlBox <- cospi.AnvlArray

#' @title Tangent of a Multiple of Pi
#' @description
#' Element-wise `tan(pi * x)`, the generic [base::tanpi()] on an anvl array.
#' Like base R's, it is exact for a whole argument and `NaN` at the half
#' integers, where the tangent has its poles.
#' @template param_x_float
#' @template return_unary_float
#' @seealso [sinpi()][sinpi.AnvlArray], [cospi()][cospi.AnvlArray], [nv_tan()]
#' @examplesIf pjrt::plugins_downloaded()
#' tanpi(nv_array(c(0, 0.25, 0.5, 1)))
#' @method tanpi AnvlArray
#' @export
#' @jit
tanpi.AnvlArray <- function(x) {
  x <- as_anvl_array(promote_to_float(x))
  denominator <- cospi(x)
  nv_ifelse(denominator == 0, NaN, sinpi(x) / denominator)
}

#' @method tanpi AnvlBox
#' @export
tanpi.AnvlBox <- tanpi.AnvlArray

#' @rdname nv_lgamma
#' @usage NULL
#' @export
lgamma.AnvlArray <- function(x) {
  nv_lgamma(x)
}

#' @export
lgamma.AnvlBox <- lgamma.AnvlArray

#' @rdname nv_digamma
#' @usage NULL
#' @export
digamma.AnvlArray <- function(x) {
  nv_digamma(x)
}

#' @export
digamma.AnvlBox <- digamma.AnvlArray

#' @rdname nv_polygamma
#' @usage NULL
#' @section Relation to base R:
#' `trigamma(x)` is `nv_polygamma(1, x)`.
#' @export
trigamma.AnvlArray <- function(x) {
  nv_polygamma(1, x)
}

#' @export
trigamma.AnvlBox <- trigamma.AnvlArray

#' @title Gamma Function
#' @description
#' Element-wise gamma function, the generic [base::gamma()] on an anvl array.
#'
#' XLA has only the log-gamma function, so `gamma()` is computed as
#' `exp(lgamma(x))` -- via Euler's reflection formula for a negative argument
#' -- and is therefore less accurate than base R's. It is `NaN` at the poles,
#' i.e. at every whole number that is not positive.
#' @template param_x_float
#' @template return_unary_float
#' @seealso [nv_lgamma()], which is what the hardware computes.
#' @examplesIf pjrt::plugins_downloaded()
#' gamma(nv_array(c(0.5, 1, 5, -1.5)))
#' @method gamma AnvlArray
#' @export
#' @jit
gamma.AnvlArray <- function(x) {
  x <- as_anvl_array(promote_to_float(x))
  positive <- nv_exp(nv_lgamma(x))
  # lgamma() is the log of the *absolute* gamma, so for a negative argument use
  # Euler's reflection formula gamma(x) * gamma(1 - x) = pi / sin(pi * x),
  # whose right-hand side is evaluated at 1 - x > 1.
  reflected <- pi / (sinpi(x) * nv_exp(nv_lgamma(1 - x)))
  out <- nv_ifelse(x < 0, reflected, positive)
  nv_ifelse((x <= 0) & (x == nv_floor(x)), NaN, out)
}

#' @method gamma AnvlBox
#' @export
gamma.AnvlBox <- gamma.AnvlArray

# Rounding ---------------------------------------------------------------------

#' @rdname nv_floor
#' @usage NULL
#' @export
floor.AnvlArray <- function(x) {
  nv_floor(x)
}

#' @export
floor.AnvlBox <- floor.AnvlArray

#' @rdname nv_ceiling
#' @usage NULL
#' @export
ceiling.AnvlArray <- function(x) {
  nv_ceiling(x)
}

#' @export
ceiling.AnvlBox <- ceiling.AnvlArray

#' @rdname nv_trunc
#' @param ... Further arguments of [nv_trunc()].
#' @export
trunc.AnvlArray <- function(x, ...) {
  nv_trunc(x, ...)
}

#' @export
trunc.AnvlBox <- trunc.AnvlArray

#' @rdname nv_round
#' @param digits (`numeric(1)`)\cr
#'   Number of digits to round to, as in [base::round()]. Must be a plain R
#'   value.
#' @param ... Further arguments of [nv_round()].
#' @section Relation to base R:
#' `round(x, digits)` is computed by scaling with `10^digits`, so it can differ
#' from [base::round()] in the last representable digit. A negative `digits`
#' rounds an integer array to a multiple of ten in base R; on an anvl array,
#' where an integer never turns into a float on its own, that is an error
#' instead.
#' @export
round.AnvlArray <- function(x, digits = 0, ...) {
  checkmate::assert_number(digits, finite = TRUE)
  if (is_intlike(x)) {
    if (digits < 0) {
      cli_abort(c(
        "{.fn round} cannot round an integer array to {digits} digits.",
        "i" = "Convert it with {.fn nv_convert} first."
      ))
    }
    # An integer array is already whole, whatever the number of digits.
    return(nv_round(x, ...))
  }
  if (digits == 0) {
    return(nv_round(x, ...))
  }
  scale <- 10^digits
  nv_round(x * scale, ...) / scale
}

#' @export
round.AnvlBox <- round.AnvlArray

#' @title Round to Significant Digits
#' @description
#' Element-wise rounding to `digits` significant digits, the generic
#' [base::signif()] on an anvl array. It is computed by rounding the mantissa,
#' so it can differ from base R's in the last representable digit.
#' @param x ([`arrayish`])\cr
#'   Input array. Must be a float array: unlike base R, an integer array is not
#'   rounded to a coarser magnitude, since that would have to turn it into a
#'   float. Convert it with [nv_convert()] if that is what you mean.
#' @param digits (`numeric(1)`)\cr
#'   Number of significant digits, as in [base::signif()]. Must be a plain R
#'   value; a value below 1 is raised to 1, like in base R.
#' @template return_unary
#' @seealso [nv_round()]
#' @examplesIf pjrt::plugins_downloaded()
#' signif(nv_array(c(123.456, -0.001234)), 3)
#' @method signif AnvlArray
#' @export
#' @jit static "digits"
signif.AnvlArray <- function(x, digits = 6) {
  checkmate::assert_number(digits, finite = TRUE)
  # Like base R, which warns and uses 1 for a smaller value.
  digits <- max(digits, 1)
  x <- as_anvl_array(assert_float_array(x))
  # Shift the value so that `digits` significant digits sit in front of the
  # decimal point, round there, and shift back.
  scale <- nv_pow(10, digits - 1 - nv_floor(nv_log10(nv_abs(x))))
  rounded <- nv_round(x * scale, method = "nearest_even") / scale
  # 0 has no magnitude, and Inf / NaN must pass through unchanged.
  nv_ifelse(nv_is_finite(x) & (x != 0), rounded, x)
}

#' @method signif AnvlBox
#' @export
signif.AnvlBox <- signif.AnvlArray

# Cumulative generics ----------------------------------------------------------

#' @rdname nv_cumsum
#' @usage NULL
#' @export
cumsum.AnvlArray <- function(x) {
  nv_cumsum(x)
}

#' @export
cumsum.AnvlBox <- cumsum.AnvlArray

#' @rdname nv_cumprod
#' @usage NULL
#' @export
cumprod.AnvlArray <- function(x) {
  nv_cumprod(x)
}

#' @export
cumprod.AnvlBox <- cumprod.AnvlArray

#' @rdname nv_cummax
#' @usage NULL
#' @export
cummax.AnvlArray <- function(x) {
  nv_cummax(x)
}

#' @export
cummax.AnvlBox <- cummax.AnvlArray

#' @rdname nv_cummin
#' @usage NULL
#' @export
cummin.AnvlArray <- function(x) {
  nv_cummin(x)
}

#' @export
cummin.AnvlBox <- cummin.AnvlArray

# Summary generics -------------------------------------------------------------

# The shared part of base R's Summary generics: every unnamed argument is data,
# so `sum(x, y)` sums both, and each one is reduced over all its axes before the
# results are combined element-wise. Named arguments are options of the
# underlying `nv_reduce_*()` (e.g. `axes = 1L`); an unsupported one errors there
# as an unused argument.
#
# `reduce` reduces one array with those options, `r_reduce` reduces a plain R
# value in R -- so that it enters as a literal and does not commit to a data
# type of its own -- and `combine` joins two reduced arguments.
summary_generic <- function(op, args, reduce, r_reduce, combine) {
  named <- nzchar(names(args) %||% rep("", length(args)))
  data <- args[!named]
  opts <- args[named]
  if (length(data) > 1L && length(opts) > 0L) {
    cli_abort(c(
      "{.fn {op}} cannot combine several data arguments with {.arg {names(opts)[1L]}}.",
      "i" = "Reduce the arrays one at a time, e.g. {.code {op}(x, axes = 1L)}."
    ))
  }
  parts <- lapply(data, function(z) {
    if (is_arrayish(z, convert_ok = FALSE)) reduce(z, opts) else r_reduce(z)
  })
  Reduce(combine, parts)
}

#' @rdname nv_reduce_sum
#' @usage NULL
#' @section Relation to base R:
#' `sum()` reduces over all axes and, like [base::sum()], takes several data
#' arguments: `sum(x, y)` is the sum of both arrays. `na.rm` becomes `nan_rm`.
#' Beyond base R, named arguments are passed on, so `sum(x, axes = 1L)` reduces
#' a single axis -- but only when `x` is the only data argument.
#' @export
sum.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic(
    "sum",
    list(...),
    function(z, opts) do.call(nv_reduce_sum, c(list(z), opts, list(nan_rm = na.rm))),
    function(z) sum(z, na.rm = na.rm),
    nv_add
  )
}

#' @export
sum.AnvlBox <- sum.AnvlArray

#' @rdname nv_reduce_prod
#' @usage NULL
#' @section Relation to base R:
#' `prod()` reduces over all axes and, like [base::prod()], takes several data
#' arguments: `prod(x, y)` is the product of both arrays. `na.rm` becomes
#' `nan_rm`. Beyond base R, named arguments are passed on, so
#' `prod(x, axes = 1L)` reduces a single axis -- but only when `x` is the only
#' data argument.
#' @export
prod.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic(
    "prod",
    list(...),
    function(z, opts) do.call(nv_reduce_prod, c(list(z), opts, list(nan_rm = na.rm))),
    function(z) prod(z, na.rm = na.rm),
    nv_mul
  )
}

#' @export
prod.AnvlBox <- prod.AnvlArray

#' @rdname nv_reduce_max
#' @usage NULL
#' @section Relation to base R:
#' `max()` reduces over all axes and, like [base::max()], takes several data
#' arguments: `max(x, y)` is the largest element of both arrays. `na.rm`
#' becomes `nan_rm`. Beyond base R, named arguments are passed on, so
#' `max(x, axes = 1L)` reduces a single axis -- but only when `x` is the only
#' data argument.
#' @export
max.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic(
    "max",
    list(...),
    function(z, opts) do.call(nv_reduce_max, c(list(z), opts, list(nan_rm = na.rm))),
    function(z) max(z, na.rm = na.rm),
    nv_max
  )
}

#' @export
max.AnvlBox <- max.AnvlArray

#' @rdname nv_reduce_min
#' @usage NULL
#' @section Relation to base R:
#' `min()` reduces over all axes and, like [base::min()], takes several data
#' arguments: `min(x, y)` is the smallest element of both arrays. `na.rm`
#' becomes `nan_rm`. Beyond base R, named arguments are passed on, so
#' `min(x, axes = 1L)` reduces a single axis -- but only when `x` is the only
#' data argument.
#' @export
min.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic(
    "min",
    list(...),
    function(z, opts) do.call(nv_reduce_min, c(list(z), opts, list(nan_rm = na.rm))),
    function(z) min(z, na.rm = na.rm),
    nv_min
  )
}

#' @export
min.AnvlBox <- min.AnvlArray

#' @title Range
#' @description
#' The smallest and the largest element, as a length-2 array -- the generic
#' [base::range()] on an anvl array. Like [base::range()] it takes several data
#' arguments and reduces all of them together.
#' @param ... ([`arrayish`])\cr
#'   Arrays to reduce, plus named arguments for [nv_reduce_min()] and
#'   [nv_reduce_max()] (e.g. `axes`), which are only accepted when there is a
#'   single array to reduce.
#' @param na.rm (`logical(1)`)\cr
#'   Forwarded to the `nan_rm` argument of [nv_reduce_min()] and
#'   [nv_reduce_max()].
#' @return [`arrayish`]
#' @seealso [nv_reduce_min()], [nv_reduce_max()]
#' @examplesIf pjrt::plugins_downloaded()
#' range(nv_array(c(3, 1, 4)))
#' @method range AnvlArray
#' @export
range.AnvlArray <- function(..., na.rm = FALSE) {
  args <- list(...)
  # Base R's range() is its min() and its max() next to each other.
  nv_concatenate(
    do.call(min.AnvlArray, c(args, list(na.rm = na.rm))),
    do.call(max.AnvlArray, c(args, list(na.rm = na.rm)))
  )
}

#' @method range AnvlBox
#' @export
range.AnvlBox <- range.AnvlArray

#' @rdname nv_reduce_any
#' @usage NULL
#' @section Relation to base R:
#' `any()` reduces over all axes and, like [base::any()], takes several data
#' arguments: `any(x, y)` asks about both arrays. It is *logical*, so -- unlike
#' base R -- a non-boolean argument is an error rather than a comparison
#' against zero. Beyond base R, named arguments are passed on, so
#' `any(x, axes = 1L)` reduces a single axis -- but only when `x` is the only
#' data argument.
#' @export
any.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic(
    "any",
    list(...),
    function(z, opts) do.call(nv_reduce_any, c(list(assert_boolean_array(z, arg = "...")), opts)),
    function(z) any(assert_boolean_array(z, arg = "..."), na.rm = na.rm),
    nv_or
  )
}

#' @export
any.AnvlBox <- any.AnvlArray

#' @rdname nv_reduce_all
#' @usage NULL
#' @section Relation to base R:
#' `all()` reduces over all axes and, like [base::all()], takes several data
#' arguments: `all(x, y)` asks about both arrays. It is *logical*, so -- unlike
#' base R -- a non-boolean argument is an error rather than a comparison
#' against zero. Beyond base R, named arguments are passed on, so
#' `all(x, axes = 1L)` reduces a single axis -- but only when `x` is the only
#' data argument.
#' @export
all.AnvlArray <- function(..., na.rm = FALSE) {
  summary_generic(
    "all",
    list(...),
    function(z, opts) do.call(nv_reduce_all, c(list(assert_boolean_array(z, arg = "...")), opts)),
    function(z) all(assert_boolean_array(z, arg = "..."), na.rm = na.rm),
    nv_and
  )
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
#' @usage NULL
#' @method is.nan AnvlArray
#' @export
is.nan.AnvlArray <- function(x) {
  nv_is_nan(x)
}

#' @method is.nan AnvlBox
#' @export
is.nan.AnvlBox <- is.nan.AnvlArray

#' @rdname nv_is_infinite
#' @usage NULL
#' @method is.infinite AnvlArray
#' @export
is.infinite.AnvlArray <- function(x) {
  nv_is_infinite(x)
}

#' @method is.infinite AnvlBox
#' @export
is.infinite.AnvlBox <- is.infinite.AnvlArray

#' @rdname nv_is_finite
#' @usage NULL
#' @method is.finite AnvlArray
#' @export
is.finite.AnvlArray <- function(x) {
  nv_is_finite(x)
}

#' @method is.finite AnvlBox
#' @export
is.finite.AnvlBox <- is.finite.AnvlArray

#' @rdname nv_transpose
#' @usage NULL
#' @section Relation to base R:
#' `t()` requires a matrix, whereas [base::t()] also transposes a vector (into
#' a one-row matrix) and reverses the axes of a higher-rank array.
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
#' @usage NULL
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
#' @usage NULL
#' @section Relation to base R:
#' `c()` concatenates scalars and 1-D arrays into a 1-D array, like
#' [base::c()] does for vectors. An array with more than one axis is an error:
#' base R would flatten it in column-major order, whereas an anvl array
#' flattens in row-major order (see the "Gotchas" vignette), so concatenate
#' those along an explicit `axis` instead.
#' @method c AnvlArray
#' @export
c.AnvlArray <- function(...) {
  args <- list(...)
  # `dim()` is the shape for an anvl array and R's own for anything else.
  ranks <- lengths(lapply(args, dim))
  if (any(ranks > 1L)) {
    cli_abort(c(
      "{.fn c} accepts only scalars and 1-D arrays.",
      "x" = "Got an array with {max(ranks)} axes.",
      "i" = "Use {.fn nv_concatenate} with an explicit {.arg axis}."
    ))
  }
  args <- lapply(args, function(x) {
    # A plain R vector has to become a 1-D R array to be arrayish at all.
    if (is.atomic(x) && is.null(dim(x)) && length(x) != 1L) array(x) else x
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
#' @usage NULL
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
#' @usage NULL
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

#' @title Shape of an Array
#' @description
#' The shape of an array, i.e. its axis sizes -- the generic [base::dim()] on an
#' anvl array, and the same thing as [shape()][tengen::shape].
#'
#' Unlike base R, it also has a value for an array with a single axis, where
#' `dim()` on an R vector is `NULL`. This is what makes [base::nrow()] and
#' [base::ncol()] report the size of axis 1 and 2 of a 1-D array.
#' @param x ([`arrayish`])\cr
#'   Input array.
#' @return (`integer()`)
#' @seealso [length()][length.AnvlArray], [shape()][tengen::shape]
#' @examplesIf pjrt::plugins_downloaded()
#' dim(nv_matrix(1:6, nrow = 2))
#' dim(nv_array(1:3))
#' @method dim AnvlArray
#' @export
dim.AnvlArray <- function(x) {
  shape(x)
}

#' @method dim AnvlBox
#' @export
dim.AnvlBox <- dim.AnvlArray

#' @title Number of Elements
#' @description
#' The number of elements of an array -- the generic [base::length()] on an
#' anvl array, i.e. the product of its axis sizes.
#' @param x ([`arrayish`])\cr
#'   Input array.
#' @return (`integer(1)`)
#' @seealso [dim()][dim.AnvlArray], [nelts()][tengen::nelts]
#' @examplesIf pjrt::plugins_downloaded()
#' length(nv_matrix(1:6, nrow = 2))
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
