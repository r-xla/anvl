# Arrays in the safetensors format: the methods `safetensors::safe_save_file()`
# writes an array with, and the constructor of the `"anvl"` framework that
# `safetensors::safe_load_file()` reads one with (registered in `.onLoad()`).

# The safetensors names of the data types an array can have.
safetensors_dtypes <- c(
  bool = "BOOL",
  i8 = "I8",
  i16 = "I16",
  i32 = "I32",
  i64 = "I64",
  ui8 = "U8",
  ui16 = "U16",
  ui32 = "U32",
  ui64 = "U64",
  f16 = "F16",
  bf16 = "BF16",
  f32 = "F32",
  f64 = "F64"
)

#' @exportS3Method safetensors::safe_tensor_buffer
safe_tensor_buffer.AnvlArray <- function(x) {
  # safetensors stores the elements in row-major order.
  as_raw(x, row_major = TRUE)
}

#' @exportS3Method safetensors::safe_tensor_meta
safe_tensor_meta.AnvlArray <- function(x) {
  list(
    # A list, so that a shape of one axis is written as a JSON array.
    shape = as.list(shape(x)),
    dtype = safetensors_dtypes[[as.character(dtype(x))]]
  )
}

# An array on the active backend from the row-major bytes `raw` of a tensor and
# its header entry `meta`.
safetensors_constructor <- function(raw, meta, device = NULL) {
  dtype <- names(safetensors_dtypes)[match(meta$dtype, safetensors_dtypes)]
  if (length(dtype) != 1L || is.na(dtype)) {
    cli_abort("The safetensors data type {.val {meta$dtype}} has no counterpart in anvl.")
  }
  nv_array(raw, shape = as.integer(meta$shape), dtype = dtype, device = device, byrow = TRUE)
}
