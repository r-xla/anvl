#' @param x,y ([`arrayish`])\cr
#'   Two inputs with a [common data type][common_dtype]. Can be
#'   <%= dtypes %>. Scalars are broadcast. An R value takes the other
#'   operand's data type when that is in its own or a higher
#'   [category][dtypes] (an R integer meeting `f64` becomes `f64`); otherwise it
#'   settles on its [default data type][default_dtypes], and the operands meet
#'   at their common data type.
