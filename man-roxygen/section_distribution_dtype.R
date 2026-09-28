#' @section Data Types:
#' `nv_d<%= dist %>()`, `nv_p<%= dist %>()` and `nv_q<%= dist %>()` compute at
#' the data type of `x`/`q`/`p`, which must be float; an R value settles on the
#' [default float][default_dtypes]. An R value for <%= params %> takes this data
#' type. An array is promoted to it, so a wider one (e.g. `f64` for an `f32`
#' `x`) is an error.
