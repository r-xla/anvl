#' @section Data Types:
#' `nv_d<%= dist %>()`, `nv_p<%= dist %>()` and `nv_q<%= dist %>()` compute at
#' a float data type and return it. If `x`/`q`/`p` is a float, they compute at
#' its data type; an R double settles on the [default float][default_dtypes].
#' An integer `x`/`q`/`p` (an array or an R integer) is converted to the widest
#' float among the arrays given for <%= params %>, or to the default float if
#' there is none. Integers beyond \eqn{2^{24}} are not all exact at `f32`, so a
#' large integer may change in that conversion.
#'
#' An R value for <%= params %> takes the computing data type. An array is
#' promoted to it, so a wider one (e.g. `f64` for an `f32` `x`) is an error.
