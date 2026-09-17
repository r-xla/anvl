#' @param lhs,rhs ([`arrayish`])\cr
#'   Two inputs. Can be any numeric data type: an integer operand is converted
#'   to the default float data type (see [`default_dtypes()`]) before the two
#'   are brought to a [common data type][common_dtype], so that common data
#'   type is always a float. Scalars are broadcast. An R double assumes the
#'   other operand's data type, falling back to its
#'   [default data type][default_dtypes]; an R integer materializes at its default
#'   and is converted like any other integer.
