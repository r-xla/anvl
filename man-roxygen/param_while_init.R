#' @param init (`named list()`)\cr
#'   Named list of initial state values. Each one becomes a parameter of the
#'   loop's sub-graphs, and those are traced before the state meets anything,
#'   so a bare R value here commits to its default data type (`f32` for a
#'   double, `i32` for an integer, `bool` for a logical) instead of taking one
#'   from the body. Where the loop carries another data type, say so:
#'   `nv_scalar(0, dtype = "f64")`, or [`nv_convert()`] for a value the caller
#'   passed in, which builds it at that data type exactly.
