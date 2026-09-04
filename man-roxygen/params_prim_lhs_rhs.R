#' @param lhs,rhs ([`arrayish`])\cr
#'   Arrayish values of <%= dtypes %>, of the same shape.
#'
#'   Both operands must have the same data type: one that already has a data
#'   type keeps it -- the primitive converts neither operand to meet the other,
#'   so two different data types are an error. An R value has no data type of
#'   its own and is built at the other operand's, as long as that data type is
#'   in its own category (a double becomes a float, an integer an integer or
#'   unsigned integer, a logical a `bool`); two R values must be of one R
#'   storage type and are built at its default data type (`f32`, `i32` or
#'   `bool`).
#'
#'   To combine different data types, convert one with [`nv_convert()`] or use
#'   the `nv_*` layer, which promotes to a common data type. See
#'   `vignette("type-promotion")`.
