#' <% .shapes <- if (exists("shapes", inherits = FALSE)) shapes else "and shape" %>
#' @param lhs,rhs ([`arrayish`])\cr
#'   Two inputs of the same data type <%= .shapes %>. Can be <%= dtypes %>. R
#'   values take the other operand's data type when it is in their
#'   [data type category][dtypes], and their
#'   [default data type][default_dtypes] when neither operand has one. An R
#'   value outside the other operand's category is an error, as are two R
#'   values of different storage types.
