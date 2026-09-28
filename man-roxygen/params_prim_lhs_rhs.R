#' <% .shapes <- if (exists("shapes", inherits = FALSE)) shapes else "and shape" %>
#' @param lhs,rhs ([`arrayish`])\cr
#'   Two inputs of the same data type <%= .shapes %>. Can be <%= dtypes %>. R
#'   values assume the other operand's data type when it is in their
#'   [data type category][dtypes], and their
#'   [default data type][default_dtypes] when neither operand has one.
