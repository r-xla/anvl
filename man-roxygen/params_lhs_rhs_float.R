#' @param lhs,rhs ([`arrayish`])\cr
#'   Left and right operand.
#'   An integer operand is converted to the default float data type (see
#'   [default_dtypes()]).
#'   Operands are [promoted to a common data type][nv_promote_to_common()].
#'   Scalars are [broadcast][nv_broadcast_scalars()] to the shape of the other operand.
