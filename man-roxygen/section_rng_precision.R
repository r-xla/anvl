#' @section Sampling Precision:
#' The sample is derived from a uniform draw at the backend's default float data
#' type (see [`default_dtypes()`]) and inherits its resolution: an `f32` uniform
#' takes one of `2^23` equally spaced values, an `f64` one of `2^52`. Where the
#' finer grid matters, raise the default float with [`with_default_dtypes()`].
