#' <% .shapes <- if (exists("shapes", inherits = FALSE)) paste0(", ", shapes) else "" %>
#' @param x ([`arrayish`])\cr
#'   One input<%= .shapes %>. Can be any numeric data type: a float is rounded and
#'   keeps its own, and an integer one is already whole and is returned
#'   unchanged. An R value materializes at its
#'   [default data type][default_dtypes] and is treated in the same way.
