# The `stablehlo::ValueType` of an `AbstractArray` (not an `RData`): a tensor
# type of its data type and its XLA shape, which is its R shape reversed (see
# R/lowering-axes.R).
at2vt <- function(x) {
  stopifnot(inherits(x, "AbstractArray"))
  stablehlo::ValueType(stablehlo::TensorType(x$dtype, Shape(rev(unclass(x$shape)))))
}

# A tensor `stablehlo::ValueType` of `dtype` and the XLA shape `shape`, for
# lowering rules that declare custom-call output types.
vt <- function(dtype, shape) {
  stablehlo::ValueType(stablehlo::TensorType(dtype = as_dtype(dtype), shape = as_shape(shape)))
}
