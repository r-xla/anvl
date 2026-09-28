# The `stablehlo::ValueType` of an `AbstractArray` (not an `RData`): a tensor
# type of its data type and shape.
at2vt <- function(x) {
  stopifnot(inherits(x, "AbstractArray"))
  stablehlo::ValueType(stablehlo::TensorType(x$dtype, x$shape))
}

# A tensor `stablehlo::ValueType` of `dtype` and `shape`, for lowering rules
# that declare custom-call output types.
vt <- function(dtype, shape) {
  if (!is_shape(shape)) {
    shape <- Shape(shape)
  }
  stablehlo::ValueType(stablehlo::TensorType(dtype = as_dtype(dtype), shape = shape))
}
