#' @param lhs,rhs ([`arrayish`])\cr
#'   Left and right operand.
#'   An integer operand is computed at the default float data type (see
#'   [default_dtypes()]), the way [base::atan2()] returns a double for integer
#'   vectors; a boolean operand is not accepted.
#'   Operands are [promoted to a common data type][nv_promote_to_common()].
#'   Scalars are [broadcast][nv_broadcast_scalars()] to the shape of the other operand.
