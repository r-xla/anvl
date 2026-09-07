#' <% .refs <- paste0("`", trimws(strsplit(dtype_args, ",")[[1L]]), "`") %>
#' <% .refs <- if (length(.refs) > 1L) paste0(paste(.refs[-length(.refs)], collapse = ", "), " and ", .refs[[length(.refs)]]) else .refs %>
#' @section Data Types:
#' <%= .refs %> are promoted to their common data type, and either of them is
#' converted to reach it: an `f32` alongside an `f64` gives `f64`, an `i32`
#' alongside an `f32` gives `f32`, a `bool` alongside an `i32` gives `i32`, and
#' an `i32` alongside a `ui32` gives `i64`. See [`common_dtype()`] for the whole
#' table. This is where the `nv_*` layer differs from the primitives, which
#' reject operands of different data types instead of converting one.
#'
#' An R value has no data type of its own. It contributes only its category to
#' the common data type -- an R double alongside an `i8` array gives `f32`,
#' where an R integer alongside that same array gives `i8` -- and it is then
#' built at the result directly rather than converted to it, so it arrives with
#' every digit it had, which is what makes `x_f64 / sqrt(2)` exact. Two R values
#' commit to the default data type of their common category (`f32`, `i32` or
#' `bool`).
#'
#' The common data type must still be one the function accepts: an R double
#' alongside an integer array promotes to a float, which the bitwise and shift
#' functions do not take.
#'
#' See `vignette("type-promotion")`.
