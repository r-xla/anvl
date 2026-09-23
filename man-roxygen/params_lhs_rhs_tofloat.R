#' @param lhs,rhs ([`arrayish`])\cr
#'   Two inputs. Can be any numeric data type: the two are first brought to a
#'   [common data type][common_dtype] and that is then converted to the default
#'   float data type (see [`default_dtypes()`]) where it is not a float
#'   already, so the result is always a float. Scalars are broadcast. An R
#'   value assumes the other operand's data type within its
#'   [data type category][dtypes], and settles on the default float when
#'   neither operand has one.
