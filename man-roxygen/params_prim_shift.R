#' @param x ([`arrayish`])\cr
#'   The array whose bits are shifted. Can be any integer data type.
#' @param shift ([`arrayish`])\cr
#'   By how many bits to shift each element of `x`. Has the same data type and
#'   shape as `x`, except that a scalar is broadcast to the shape of the other.
#'   An R value takes the other operand's data type, and its
#'   [default data type][default_dtypes] when neither operand has one.
