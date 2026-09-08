#' <% .any <- grepl("^any", dtypes) %>
#' <% .shapes <- if (exists("shapes", inherits = FALSE)) paste0(", ", shapes) else "" %>
#' @param x ([`arrayish`])\cr
#'   One input<%= .shapes %>. <%= if (.any) "Can be" else "Must be" %> <%= dtypes %>.<% if (.any) { %> An R value commits to its
#'   [default data type][default_dtypes].<% } %>
