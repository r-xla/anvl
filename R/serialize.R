#' @title Save and Read Arrays in a File
#'
#' @description
#' `nv_save()` saves a named list of arrays to a file in the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format, and
#' `nv_read()` loads them back. The data type and shape of each array are
#' restored.
#'
#' @details
#' These are convenience wrappers around [`nv_serialize()`] and
#' [`nv_unserialize()`] that open and close a file connection.
#'
#' @param arrays (named `list` of [`AnvlArray`])\cr
#'   Named list of arrays. Names must be unique.
#' @param path (`character(1)`)\cr
#'   File path to write to or read from.
#' @param device (`NULL` | `character(1)` | [device][nv_device])\cr
#'   The device of the active backend to place the loaded arrays on, given as
#'   for [`nv_array()`]. The default (`NULL`) uses [`default_device()`].
#'
#' @returns `nv_save()`: (`NULL`)\cr
#'   Invisibly.
#'
#'   `nv_read()`: (named `list` of [`AnvlArray`])
#' @seealso [nv_serialize()], [nv_unserialize()]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' x <- nv_matrix(1:6, nrow = 2)
#' x
#' path <- tempfile(fileext = ".safetensors")
#' nv_save(list(x = x), path)
#' nv_read(path)
nv_save <- function(arrays, path) {
  if (is_arrayish(arrays) || !is.list(arrays)) {
    cli_abort("{.arg arrays} must be a named list of arrays, not a single array.")
  }
  checkmate::assert_list(arrays, names = "unique", types = "AnvlArray")
  checkmate::assert_string(path)

  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  nv_serialize(arrays, con = con)
  invisible(NULL)
}

#' @rdname nv_save
#' @export
nv_read <- function(path, device = NULL) {
  checkmate::assert_string(path)
  checkmate::assert_file_exists(path)
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  nv_unserialize(con, device = device)
}

#' @title Serialize Arrays to Raw Bytes
#'
#' @description
#' `nv_serialize()` serializes a named list of arrays into the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format, and
#' `nv_unserialize()` deserializes them. The data type and shape of each array
#' are restored.
#'
#' @param arrays (named `list` of [`AnvlArray`])\cr
#'   Named list of arrays to serialize. Names must be unique.
#' @param con (`NULL` | connection | [`raw`])\cr
#'   For `nv_serialize()`, an optional connection to write to; if `NULL`
#'   (default), a raw vector is returned. For `nv_unserialize()`, a connection
#'   or raw vector to read from.
#' @param device (`NULL` | `character(1)` | [device][nv_device])\cr
#'   The device of the active backend to place the loaded arrays on, given as
#'   for [`nv_array()`]. The default (`NULL`) uses [`default_device()`].
#'
#' @returns `nv_serialize()`: ([`raw`] | `NULL`)\cr
#'   A raw vector if `con` is `NULL`, otherwise `NULL` invisibly.
#'
#'   `nv_unserialize()`: (named `list` of [`AnvlArray`])
#' @seealso [nv_save()], [nv_read()]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' x <- nv_matrix(1:6, nrow = 2)
#' x
#' raw_data <- nv_serialize(list(x = x))
#' raw_data
#' nv_unserialize(raw_data)
nv_serialize <- function(arrays, con = NULL) {
  # `assert_list(types = )` subsets `arrays` internally, and an `AnvlArray` has
  # a `[` method -- so a bare array sends the *assertion* into `nv_subset()` and
  # the caller sees a subsetting error. Say what is wrong first, as
  # `prim_sort()` does for the same trap.
  if (is_arrayish(arrays) || !is.list(arrays)) {
    cli_abort("{.arg arrays} must be a named list of arrays, not a single array.")
  }
  checkmate::assert_list(arrays, names = "unique", types = "AnvlArray")

  if (is.null(con)) {
    return(safetensors::safe_serialize(arrays))
  }
  safetensors::safe_save_file(arrays, con)
  # The connection is written for its side effect; `nv_save()` returns nothing
  # either.
  invisible(NULL)
}

#' @rdname nv_serialize
#' @export
nv_unserialize <- function(con, device = NULL) {
  arrays <- safetensors::safe_load_file(con, framework = "anvl", device = device)
  # Drop the header metadata safetensors attaches.
  attributes(arrays) <- list(names = names(arrays))
  arrays
}

# anvl's data types by their name in the safetensors format.
safetensors_dtypes <- c(
  BOOL = "bool",
  I8 = "i8",
  I16 = "i16",
  I32 = "i32",
  I64 = "i64",
  U8 = "ui8",
  U16 = "ui16",
  U32 = "ui32",
  U64 = "ui64",
  F32 = "f32",
  F64 = "f64"
)

anvl_dtype_to_safetensors <- function(dtype) {
  dtype <- as.character(dtype)
  name <- names(safetensors_dtypes)[match(dtype, safetensors_dtypes)]
  if (is.na(name)) {
    cli_abort("Data type {.val {dtype}} cannot be saved in the safetensors format.")
  }
  name
}

safetensors_dtype_to_anvl <- function(name) {
  if (!name %in% names(safetensors_dtypes)) {
    cli_abort("Unsupported safetensors data type {.val {name}}.")
  }
  safetensors_dtypes[[name]]
}

#' @exportS3Method safetensors::safe_tensor_meta
safe_tensor_meta.AnvlArray <- function(x) {
  # A list, so that a shape of one axis is still written as a JSON array.
  list(shape = as.list(shape(x)), dtype = anvl_dtype_to_safetensors(dtype(x)))
}

#' @exportS3Method safetensors::safe_tensor_buffer
safe_tensor_buffer.AnvlArray <- function(x) {
  as_raw(x, row_major = TRUE)
}

# The constructor of the safetensors framework "anvl" (registered in
# `.onLoad()`): builds an array of the active backend from the row-major bytes
# of a tensor.
anvl_array_from_safetensors <- function(raw, meta, device = NULL) {
  nv_array(
    raw,
    dtype = safetensors_dtype_to_anvl(meta$dtype),
    shape = as.integer(meta$shape),
    device = device,
    byrow = TRUE
  )
}
