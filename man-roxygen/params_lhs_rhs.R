#' @param lhs,rhs ([`arrayish`])\cr
#'   Two inputs with a [common data type][common_dtype]. Can be
#'   <%= dtypes %>. Scalars are broadcast, and R values assume the other
#'   operand's data type within their [data type category][dtypes],
#'   otherwise falling back to their [default data type][default_dtypes] and
#'   being converted to the common data type.
