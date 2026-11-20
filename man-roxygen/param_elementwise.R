#' @param elementwise (`logical(1)`)\cr
#'   Whether the primitive is elementwise: each element of its single result is
#'   computed from the elements at the same position of its operands alone,
#'   where a scalar operand is broadcast, at the operands' data type or as a
#'   boolean. Its derivative is then diagonal, and its `pushforward` rule is
#'   derived from its `pullback` rule.
