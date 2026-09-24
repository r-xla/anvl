#' Convert AbstractArray to ValueType
#' @description
#' Convert an [`AbstractArray`] to a [`stablehlo::ValueType`].
#' @param x ([`AbstractArray`])\cr
#'   The abstract array. Must not be an [`RData`].
#' @return ([`stablehlo::ValueType`])\cr
#'   A tensor type of `x`'s data type and shape.
#' @examples
#' at2vt(nv_aval("f32", c(2L, 3L)))
#' @export
at2vt <- function(x) {
  stopifnot(inherits(x, "AbstractArray"))
  stablehlo::ValueType(stablehlo::TensorType(x$dtype, x$shape))
}

#' Construct a stablehlo ValueType
#' @description
#' Shorthand for building a tensor [`stablehlo::ValueType`] from a dtype
#' and shape — convenient inside stablehlo lowering rules that need to
#' declare custom-call output types or similar.
#' @param dtype (`character(1)` | [`tengen::DataType`])\cr
#'   The data type.
#' @param shape (`integer()` | [`stablehlo::Shape`])\cr
#'   The shape.
#' @return ([`stablehlo::ValueType`])\cr
#'   A tensor type of the given data type and shape.
#' @examples
#' vt("f32", c(2L, 3L))
#' @export
vt <- function(dtype, shape) {
  if (!is_shape(shape)) {
    shape <- Shape(shape)
  }
  stablehlo::ValueType(stablehlo::TensorType(dtype = as_dtype(dtype), shape = shape))
}
