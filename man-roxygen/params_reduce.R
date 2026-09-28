#' <% .all <- if (exists("axes_all", inherits = FALSE)) paste0(" ", axes_all) else "" %>
#' @param axes (`integer()`<%= if (nzchar(.all)) " | `NULL`" else "" %>)\cr
#'   Axes to reduce over. Negative values count from the end, i.e. `-1` refers
#'   to the last axis.<%= .all %>
#' @param drop (`logical(1)`)\cr
#'   Whether to drop the reduced axes: removed from the output shape if `TRUE`,
#'   set to 1 if `FALSE`.
