#' <% p <- primitive_env[[primitive_id]]; implemented <- Filter(function(r) !is.null(p[[r]]), globals$interpretation_rules) %>
#' <% if (length(implemented) > 0) { %>
#' @section Implemented Rules:
#' <%= paste0("- `", implemented, "`", collapse = "\n#' ") %>
#' <% } %>
