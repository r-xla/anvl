# The safetensors format itself is tested in {safetensors} and {pjrt}; these
# tests cover the `AnvlArray` side of it.
skip_if_not_installed("safetensors")

# A safetensors file as another framework writes it: an 8-byte little-endian
# header length, the JSON header, then the packed row-major elements.
safetensors_bytes <- function(header, payload) {
  c(writeBin(nchar(header), raw(), size = 8L, endian = "little"), charToRaw(header), payload)
}

# `safe_load_file()` also attaches the file's header; `[` keeps only the names.
safetensors_roundtrip <- function(arrays, ...) {
  path <- withr::local_tempfile(fileext = ".safetensors")
  safetensors::safe_save_file(arrays, path)
  safetensors::safe_load_file(path, framework = "anvl", ...)[names(arrays)]
}

describe("safetensors::safe_save_file() and safetensors::safe_load_file()", {
  it("round-trip a named list of arrays", {
    lst <- list(
      x = nv_matrix(rnorm(12L), nrow = 3L),
      scalar = nv_scalar(1.5),
      i = nv_array(1:3)
    )
    expect_equal(safetensors_roundtrip(lst), lst)
  })

  it("round-trip every data type", {
    dtypes <- c("bool", "i8", "i16", "i32", "i64", "ui8", "ui16", "ui32", "ui64", "f16", "bf16", "f32", "f64")
    data <- array(c(0L, 1L, 1L, 0L, 1L, 1L), c(2L, 3L))
    lst <- lapply(dtypes, function(dt) nv_array(data, dtype = dt))
    names(lst) <- dtypes
    expect_equal(safetensors_roundtrip(lst), lst)
  })

  it("load onto the default device", {
    skip_if(!is_cpu())
    local_default_device("cpu:1")
    expect_equal(device(safetensors_roundtrip(list(x = nv_array(1:3)))$x), nv_device("cpu:1"))
  })

  it("load onto the device given", {
    skip_if(!is_cpu())
    local_default_device("cpu:0")
    expect_equal(device(safetensors_roundtrip(list(x = nv_array(1:3)), device = "cpu:1")$x), nv_device("cpu:1"))
  })

  it("round-trip arrays of the quickr backend", {
    skip_if_no_quickr()
    local_backend("quickr")
    lst <- list(
      f = nv_matrix(c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5), nrow = 2L),
      i = nv_matrix(1:6, nrow = 3L),
      b = nv_array(c(TRUE, FALSE, TRUE))
    )
    reloaded <- safetensors_roundtrip(lst)
    expect_equal(backend(reloaded$f), "quickr")
    expect_equal(reloaded, lst)
    expect_equal(device(safetensors_roundtrip(lst, device = "cpu")$f), quickr_device("cpu"))
  })
})

describe("the safetensors methods for AnvlArray", {
  it("write the elements in row-major order", {
    x <- nv_array(matrix(1:6, nrow = 2L), dtype = "i32")
    expect_identical(
      safetensors::safe_tensor_buffer(x),
      writeBin(c(1L, 3L, 5L, 2L, 4L, 6L), raw(), size = 4L, endian = "little")
    )
    expect_identical(safetensors::safe_tensor_meta(x), list(shape = list(2L, 3L), dtype = "I32"))
  })

  it("write a file the pjrt framework reads", {
    x <- nv_array(matrix(rnorm(6L), nrow = 2L), dtype = "f64")
    buf <- safetensors::safe_load_file(safetensors::safe_serialize(list(x = x)), framework = "pjrt")$x
    expect_identical(as_array(buf), as_array(x))
  })

  it("read a file another framework wrote", {
    header <- '{"x":{"dtype":"I32","shape":[2,3],"data_offsets":[0,24]}}'
    bytes <- safetensors_bytes(header, writeBin(1:6, raw(), size = 4L, endian = "little"))
    x <- safetensors::safe_load_file(bytes, framework = "anvl")$x
    expect_dtype(x, "i32")
    expect_identical(as_array(x), matrix(1:6, nrow = 2L, byrow = TRUE))
  })

  it("read a scalar", {
    header <- '{"s":{"dtype":"F64","shape":[],"data_offsets":[0,8]}}'
    bytes <- safetensors_bytes(header, writeBin(2.5, raw(), size = 8L, endian = "little"))
    s <- safetensors::safe_load_file(bytes, framework = "anvl")$s
    expect_shape(s, integer())
    expect_identical(as_array(s), 2.5)
  })

  it("refuse a data type anvl does not have", {
    header <- '{"x":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}'
    bytes <- safetensors_bytes(header, as.raw(0L))
    expect_error(safetensors::safe_load_file(bytes, framework = "anvl"), "no counterpart in anvl")
    header <- '{"x":{"shape":[1],"data_offsets":[0,1]}}'
    bytes <- safetensors_bytes(header, as.raw(0L))
    expect_error(safetensors::safe_load_file(bytes, framework = "anvl"), "no counterpart in anvl")
  })

  it("read a file onto the quickr backend", {
    skip_if_no_quickr()
    x <- nv_array(matrix(rnorm(6L), nrow = 2L), dtype = "f64")
    bytes <- safetensors::safe_serialize(list(x = x))
    local_backend("quickr")
    y <- safetensors::safe_load_file(bytes, framework = "anvl")$x
    expect_equal(backend(y), "quickr")
    expect_identical(as_array(y), as_array(x))
  })

  it("refuse a data type on the quickr backend that it does not support", {
    skip_if_no_quickr()
    bytes <- safetensors::safe_serialize(list(x = nv_array(1:3, dtype = "f32")))
    local_backend("quickr")
    expect_error(safetensors::safe_load_file(bytes, framework = "anvl"), "Unsupported dtype for the .*quickr")
  })
})
