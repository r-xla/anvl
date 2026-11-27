#' @section Non-Integer Values:
#' `x` and `q` are counts, but any value is accepted, and one that is not a
#' whole number is treated as in base R, without its warnings:
#' * `nv_d<%= dist %>()` takes an `x` within a relative `1e-9` of a whole
#'   number, that is with
#'   \eqn{|x - \mathrm{round}(x)| \le 10^{-9} \max(|x|, 1)}{|x - round(x)| <= 1e-9 * max(|x|, 1)},
#'   as that whole number. Any other `x` has probability zero.
#' * `nv_p<%= dist %>()` rounds `q` down, except that a `q` within `1e-7` below
#'   a whole number counts as that number.
#' * A negative `x` or `q` is below the support even where it is within the
#'   tolerance of zero: such an `x` has probability zero, and such a `q` has
#'   \eqn{P(X \le q) = 0}.
#'
#' Finite quantiles from `nv_q<%= dist %>()` are always whole numbers. Since
#' `x` and `q` are used only as whole numbers, their gradient is zero wherever
#' a gradient can be taken.
