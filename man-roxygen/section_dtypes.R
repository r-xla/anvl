#' <% .refs <- paste0("`", trimws(strsplit(dtype_args, ",")[[1L]]), "`") %>
#' <% .refs <- if (length(.refs) > 1L) paste0(paste(.refs[-length(.refs)], collapse = ", "), " and ", .refs[[length(.refs)]]) else .refs %>
#' @section Data Types:
#' <%= .refs %> must have the same data type, which the primitive does not
#' negotiate: an operand that already has a data type keeps it, and operands
#' with different data types are an error. Convert one explicitly with
#' [`nv_convert()`], or use the `nv_*` layer, which promotes to a common data
#' type.
#'
#' An R value has no data type of its own and is built at the one the other
#' operands have, as long as that data type is in its own category: a double
#' becomes a float, an integer an integer or unsigned integer, a logical a
#' `bool`. If every operand is an R value, they must be of one R storage type
#' and are built at its default data type (`f32`, `i32` or `bool`).
#'
#' See `vignette("type-promotion")`.
