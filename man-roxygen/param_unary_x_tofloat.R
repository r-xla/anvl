#' <% .shapes <- if (exists("shapes", inherits = FALSE)) paste0(", ", shapes) else "" %>
#' @param x ([`arrayish`])\cr
#'   One input<%= .shapes %>. Can be any numeric data type: a float keeps its
#'   own, and an integer one is converted to the default float data type (see
#'   [`default_dtypes()`]). An R value materializes at its
#'   [default data type][default_dtypes] and is converted in the same way.
