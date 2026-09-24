#' Convert AbstractArray to ValueType
#' @description
#' Convert an `AbstractArray` to a `ValueType`.
#' @param x (`AbstractArray`)
#' @return (`ValueType`)
#' @export
at2vt <- function(x) {
  stopifnot(inherits(x, "AbstractArray"))
  stablehlo::ValueType(stablehlo::TensorType(x$dtype, x$shape))
}

# The inverse of at2vt(), for the output types a custom call declares.
vt2at <- function(x) {
  stopifnot(inherits(x, "ValueType"), inherits(x$type, "TensorType"))
  AbstractArray(x$type$dtype, x$type$shape)
}

#' Construct a stablehlo ValueType
#' @description
#' Shorthand for building a tensor [`stablehlo::ValueType`] from a dtype
#' and shape — convenient inside stablehlo lowering rules that need to
#' declare custom-call output types or similar.
#' @param dtype A dtype (string or [`tengen::DataType`]).
#' @param shape An integer vector or [`stablehlo::Shape`].
#' @return (`ValueType`)
#' @export
vt <- function(dtype, shape) {
  if (!is_shape(shape)) {
    shape <- Shape(shape)
  }
  stablehlo::ValueType(stablehlo::TensorType(dtype = as_dtype(dtype), shape = shape))
}
