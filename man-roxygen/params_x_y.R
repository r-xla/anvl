#' @param x,y ([`arrayish`])\cr
#'   Two inputs with a [common data type][common_dtype]. Can be
#'   <%= dtypes %>. Scalars are broadcast. An R value takes the other
#'   operand's data type when that is in its own or a higher
#'   [category][dtypes]. Otherwise it
#'   settles on its [default data type][default_dtypes], and the operands meet
#'   at their common data type.
