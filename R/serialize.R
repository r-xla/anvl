#' @title Save arrays to a file
#'
#' @description
#' Saves a named list of arrays to a file in the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format.
#'
#' @details
#' This is a convenience wrapper around [`nv_serialize()`] that opens and closes
#' a file connection.
#'
#' @param arrays (named `list` of [`AnvlArray`])\cr
#'   Named list of arrays to save. Names must be unique.
#' @param path (`character(1)`)\cr
#'   File path to write to.
#'
#' @returns `NULL` (invisibly).
#' @seealso [nv_read()], [nv_serialize()], [nv_unserialize()]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' x <- nv_matrix(1:6, nrow = 2)
#' x
#' path <- tempfile(fileext = ".safetensors")
#' nv_save(list(x = x), path)
#' nv_read(path)
nv_save <- function(arrays, path) {
  checkmate::assert_list(arrays, names = "unique", types = "AnvlArray")
  checkmate::assert_string(path)

  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  nv_serialize(arrays, con = con)
  invisible(NULL)
}

#' @title Read arrays from a file
#'
#' @description
#' Loads arrays from a file in the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format.
#'
#' @details
#' This is a convenience wrapper around [`nv_unserialize()`] that opens and
#' closes a file connection.
#'
#' @param path (`character(1)`)\cr
#'   Path to the safetensors file.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device on which to place the loaded arrays (`"cpu"`, `"cuda"`, ...).
#'   Default is to use the CPU.
#' @param backend (`character(1)`)\cr
#'   Backend for the loaded arrays.
#'   Defaults to `default_backend()`.
#'
#' @returns Named `list` of [`AnvlArray`] objects.
#' @seealso [nv_save()], [nv_serialize()], [nv_unserialize()]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' x <- nv_matrix(1:6, nrow = 2)
#' x
#' path <- tempfile(fileext = ".safetensors")
#' nv_save(list(x = x), path)
#' nv_read(path)
nv_read <- function(path, device = NULL, backend = default_backend()) {
  checkmate::assert_string(path)
  checkmate::assert_file_exists(path)
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  nv_unserialize(con, device = device, backend = backend)
}

#' @title Serialize arrays to raw bytes
#'
#' @description
#' Serializes a named list of arrays into the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format.
#'
#' @details
#' The ambiguity of the arrays is stored in the metadata and preserved in write-read roundtrips.
#'
#' @param arrays (named `list` of [`AnvlArray`])\cr
#'   Named list of arrays to serialize. Names must be unique.
#' @param con (`NULL` | connection)\cr
#'   An optional connection to write to.
#'   If `NULL` (default), a raw vector is returned.
#'
#' @returns A [`raw`] vector if `con` is `NULL`, otherwise `NULL` (invisibly).
#' @seealso [nv_unserialize()], [nv_save()], [nv_read()]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' x <- nv_matrix(1:6, nrow = 2)
#' x
#' raw_data <- nv_serialize(list(x = x))
#' raw_data
#' nv_unserialize(raw_data)
nv_serialize <- function(arrays, con = NULL) {
  checkmate::assert_list(arrays, names = "unique", types = "AnvlArray")

  # Extract ambiguity information to store in metadata
  ambiguity_info <- lapply(arrays, function(t) if (ambiguous(t)) TRUE else FALSE)
  names(ambiguity_info) <- names(arrays)

  # Serialize using base R serialize() as raw bytes, encoded as hex string
  raw_bytes <- serialize(ambiguity_info, connection = NULL)
  hex_string <- paste(sprintf("%02x", as.integer(raw_bytes)), collapse = "")
  metadata <- list(`__ambiguity_info__` = hex_string)

  # TODO(hack): do this properly
  arrays_unwrapped <- lapply(arrays, function(x) {
    buf <- unwrap_if_array(x)
    if (inherits(buf, "PJRTBuffer")) {
      return(buf)
    }
    pjrt_buffer(buf, dtype = as.character(dtype(x)), shape = shape(x))
  })

  if (is.null(con)) {
    safetensors::safe_serialize(arrays_unwrapped, metadata = metadata)
  } else {
    safetensors::safe_save_file(arrays_unwrapped, con, metadata = metadata)
  }
}

#' @title Deserialize arrays from raw bytes
#'
#' @description
#' Deserializes arrays from the
#' [safetensors](https://huggingface.co/docs/safetensors/index) format.
#'
#' @details
#' The data type, shape, and [ambiguity][ambiguous()] of each array are
#' restored from the serialized data.
#'
#' @param con (connection | [`raw`])\cr
#'   A connection or raw vector to read from.
#' @param device (`NULL` | `character(1)` | [`PJRTDevice`][pjrt::pjrt_device])\cr
#'   The device on which to place the loaded arrays (`"cpu"`, `"cuda"`, ...).
#'   Default is to use the CPU.
#' @param backend (`character(1)`)\cr
#'   Backend for the loaded arrays.
#'   Defaults to `default_backend()`.
#'
#' @returns Named `list` of [`AnvlArray`] objects.
#' @seealso [nv_serialize()], [nv_save()], [nv_read()]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' x <- nv_matrix(1:6, nrow = 2)
#' x
#' raw_data <- nv_serialize(list(x = x))
#' raw_data
#' nv_unserialize(raw_data)
nv_unserialize <- function(con, device = NULL, backend = default_backend()) {
  # TODO: don't convert to pjrt first
  result <- safetensors::safe_load_file(con, framework = "pjrt", device = device)

  # Extract metadata to restore ambiguity information
  # Metadata is nested under __metadata__ key
  metadata_attr <- attr(result, "metadata", exact = TRUE)
  metadata <- if (!is.null(metadata_attr)) metadata_attr$`__metadata__` else NULL

  # Parse ambiguity info using base R (unserialize from hex string)
  ambiguity_info <- if (!is.null(metadata) && !is.null(metadata$`__ambiguity_info__`)) {
    hex_string <- metadata$`__ambiguity_info__`
    # Convert hex string back to raw bytes
    hex_pairs <- strsplit(hex_string, "(?<=..)", perl = TRUE)[[1]]
    raw_bytes <- as.raw(strtoi(hex_pairs, base = 16L))
    unserialize(raw_bytes)
  } else {
    NULL
  }

  backend <- assert_backend(backend)
  result_wrapped <- lapply(names(result), function(name) {
    buf <- result[[name]]
    is_ambiguous <- if (!is.null(ambiguity_info) && !is.null(ambiguity_info[[name]])) {
      isTRUE(ambiguity_info[[name]])
    } else {
      FALSE
    }
    if (backend == "pjrt") {
      nv_array(buf, ambiguous = is_ambiguous, backend = "pjrt")
    } else {
      nv_array(
        tengen::as_array(buf),
        dtype = as.character(pjrt::elt_type(buf)),
        shape = tengen::shape(buf),
        ambiguous = is_ambiguous,
        backend = backend
      )
    }
  })
  names(result_wrapped) <- names(result)
  result_wrapped
}
