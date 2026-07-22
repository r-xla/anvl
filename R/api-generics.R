#' @include api.R

#' @export
Ops.AnvlArray <- function(e1, e2) {
  switch(
    .Generic, # nolint
    "+" = nv_add(e1, e2),
    "-" = {
      if (missing(e2)) {
        nv_negate(e1)
      } else {
        nv_sub(e1, e2)
      }
    },
    "*" = nv_mul(e1, e2),
    "/" = nv_div(e1, e2),
    "^" = nv_pow(e1, e2),
    "%%" = nv_mod(e1, e2),
    "==" = nv_eq(e1, e2),
    "!=" = nv_ne(e1, e2),
    ">" = nv_gt(e1, e2),
    ">=" = nv_ge(e1, e2),
    "<" = nv_lt(e1, e2),
    "<=" = nv_le(e1, e2),
    "!" = nv_not(e1),
    "&" = nv_and(e1, e2),
    "|" = nv_or(e1, e2)
  )
}

#' @export
Ops.AnvlBox <- Ops.AnvlArray

#' @export
matrixOps.AnvlArray <- function(x, y) {
  switch(
    .Generic, # nolint
    "%*%" = nv_matmul(x, y)
  )
}

#' @export
matrixOps.AnvlBox <- matrixOps.AnvlArray

#' @export
Math.AnvlArray <- function(x, ...) {
  # Forward `...` so extras like `log(x, base = 2)` surface as
  # "unused argument" errors from the underlying nv_* function.
  switch(
    .Generic, # nolint
    "abs" = nv_abs(x, ...),
    "exp" = nv_exp(x, ...),
    "sqrt" = nv_sqrt(x, ...),
    "log" = nv_log(x, ...),
    "log2" = nv_log2(x, ...),
    "log10" = nv_log10(x, ...),
    "tanh" = nv_tanh(x, ...),
    "tan" = nv_tan(x, ...),
    "cos" = nv_cos(x, ...),
    "sin" = nv_sin(x, ...),
    "acos" = nv_acos(x, ...),
    "acosh" = nv_acosh(x, ...),
    "asin" = nv_asin(x, ...),
    "asinh" = nv_asinh(x, ...),
    "atan" = nv_atan(x, ...),
    "atanh" = nv_atanh(x, ...),
    "cosh" = nv_cosh(x, ...),
    "sinh" = nv_sinh(x, ...),
    "digamma" = nv_digamma(x, ...),
    "lgamma" = nv_lgamma(x, ...),
    "trigamma" = nv_polygamma(1, x, ...),
    "floor" = nv_floor(x, ...),
    "ceiling" = nv_ceiling(x, ...),
    "trunc" = nv_trunc(x, ...),
    "sign" = nv_sign(x, ...),
    "expm1" = nv_expm1(x, ...),
    "log1p" = nv_log1p(x, ...),
    "round" = nv_round(x, method = "nearest_even", ...),
    "cumsum" = nv_cumsum(x, ...),
    "cumprod" = nv_cumprod(x, ...),
    "cummax" = nv_cummax(x, ...),
    "cummin" = nv_cummin(x, ...),
    cli_abort("invalid method: {(.Generic)}")
  )
}

#' @export
Math.AnvlBox <- Math.AnvlArray


#' @export
Summary.AnvlArray <- function(x, ..., na.rm = FALSE) {
  # Forward `na.rm` to the underlying nv_reduce_*'s `nan_rm` arg and `...`
  # for supported extras (e.g. `sum(x, dims = 1L)`); unsupported extras error
  # there as unused args.
  switch(
    .Generic, # nolint
    "max" = nv_reduce_max(x, ..., nan_rm = na.rm),
    "min" = nv_reduce_min(x, ..., nan_rm = na.rm),
    "range" = nv_concatenate(
      nv_reduce_min(x, ..., nan_rm = na.rm),
      nv_reduce_max(x, ..., nan_rm = na.rm)
    ),
    "prod" = nv_reduce_prod(x, ..., nan_rm = na.rm),
    "sum" = nv_reduce_sum(x, ..., nan_rm = na.rm),
    # no na.rm because we only support booleans which can't have NaNs
    "any" = nv_reduce_any(x, ...),
    "all" = nv_reduce_all(x, ...),
    cli_abort("invalid method: {(.Generic)}")
  )
}

#' @export
Summary.AnvlBox <- Summary.AnvlArray

#' @rdname nv_mean
#' @param trim Currently not supported.
#' @param na.rm Forwarded to [nv_mean()]'s `nan_rm` argument.
#' @param ... No additional arguments.
#' @method mean AnvlArray
#' @export
mean.AnvlArray <- function(x, trim = 0, na.rm = FALSE, ..., dims = NULL, drop = TRUE) {
  if (!identical(trim, 0)) {
    cli_abort("{.arg trim} is not supported by {.fn mean} for anvl arrays.")
  }
  nv_mean(x, ..., dims = dims, drop = drop, nan_rm = na.rm)
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
#' Permutes the dimensions of an array. You can also use `t()` for matrices.
#' @param permutation (`integer()` | `NULL`)\cr
#'   New ordering of dimensions. If `NULL` (default), reverses the dimensions.
#'   Negative values count from the end, i.e. `-1` refers to the last dimension.
#' @return [`arrayish`]\cr
#'   Has the same data type as `x` and shape `nv_shape(x)[permutation]`.
#' @seealso [prim_transpose()] for the underlying primitive.
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_matrix(1:6, nrow = 2)
#' t(x)
#' @method t AnvlArray
#' @export
t.AnvlArray <- function(x) {
  nd <- ndims(x)
  if (nd != 2L) {
    cli_abort("{.fn t} requires a 2-D array, but got a {nd}-D array.")
  }
  nv_transpose(x)
}

#' @method t AnvlBox
#' @export
t.AnvlBox <- t.AnvlArray

#' @rdname nv_median
#' @param na.rm Forwarded to [nv_median()]'s `nan_rm` argument.
#' @param ... No additional arguments.
#' @method median AnvlArray
#' @export
median.AnvlArray <- function(x, na.rm = FALSE, ..., dim = NULL, interpolation = "linear") {
  rlang::check_dots_empty()
  nv_median(x, dim = dim, interpolation = interpolation, nan_rm = na.rm)
}

#' @method median AnvlBox
#' @export
median.AnvlBox <- median.AnvlArray

#' @rdname nv_sort
#' @param decreasing (`logical(1)`)\cr If `TRUE`, sort in decreasing order.
#' @param ... No additional arguments.
#' @method sort AnvlArray
#' @export
sort.AnvlArray <- function(x, decreasing = FALSE, ..., dim = NULL) {
  nv_sort(x, decreasing = decreasing, ..., dim = dim)
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
  rank <- length(shape_abstract(x))
  if (n_args > rank) {
    cli_abort("Too many subset specifications: got {n_args}, expected at most {rank}")
  }
  quos <- rlang::enquos(...)
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
  rank <- length(shape_abstract(x))
  if (n_args > rank) {
    cli_abort("Too many subset specifications: got {n_args}, expected at most {rank}")
  }
  quos <- rlang::enquos(...)
  rlang::inject(nv_subset_assign(x, !!!quos, value = value))
}

#' @method [<- AnvlBox
#' @export
`[<-.AnvlBox` <- `[<-.AnvlArray`

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
