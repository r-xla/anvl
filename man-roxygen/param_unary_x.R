#' <% .any <- grepl("^any", dtypes) %>
#' @param x ([`arrayish`])\cr
#'   One input. <%= if (.any) "Can be" else "Must be" %> <%= dtypes %>.<% if (.any) { %> An R value commits to its
#'   [default data type][default_dtypes].<% } %>
