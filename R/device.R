#' @title Get the default device
#' @description
#' Returns the default device of the active backend.
#' For the `"pjrt"` backend, the default device is configured by the `PJRT_PLATFORM`
#' environment variable (defaulting to `"cpu"`). Other backends (e.g. `"quickr"`)
#' only support CPU.
#' @param backend (`NULL` | `character(1)`)\cr
#'   Backend. Defaults to [`active_backend()`] when `NULL`.
#' @return A backend-specific device object.
#' @seealso [`nv_device()`], [`active_backend()`]
#' @export
default_device <- function(backend = NULL) {
  backend <- backend %||% active_backend()
  platform <- if (backend == "pjrt") Sys.getenv("PJRT_PLATFORM", "cpu") else "cpu"
  backend_device(platform, backend)
}

#' @title Create a Device
#' @description
#' Constructs a backend-specific device object for the active backend
#' ([`active_backend()`]).
#'
#' A device identifies a compute resources, such as CPU, or a specific GPU.
#' It is relevant for data allocation (e.g. via [nv_array()]) but also compilation ([jit]).
#' A device belongs to the active backend ([`active_backend()`]); a device
#' object of another backend is an error.
#'
#' @param x (`character(1)` | device object)\cr
#'   Identifier for the device (e.g. `"cpu"`, `"cuda"`, `"cuda:<n>"`),
#'   or an existing device object of the active backend (returned as-is).
#' @return A backend-specific device object (e.g. `PJRTDevice` for `"pjrt"`,
#'   [`quickr_device`] for `"quickr"`).
#' @seealso [`backend()`], [`AnvlBackend()`], [`active_backend()`].
#' @examplesIf pjrt::plugins_downloaded()
#' # Create CPU device for the active backend
#' nv_device("cpu")
#' # Create CPU device for the quickr backend:
#' with_backend("quickr", nv_device("cpu"))
#' # Pass through an existing device:
#' dev <- nv_device("cpu")
#' identical(nv_device(dev), dev)
#' @export
nv_device <- function(x) {
  backend_device(x, active_backend())
}

# `x` as a device of `backend`: a device object is checked to belong to it, a
# string is looked up on it.
backend_device <- function(x, backend) {
  if (is_device(x)) {
    check_device_backend(x, backend)
    return(x)
  }
  backend <- assert_backend(backend)
  globals$backends[[backend]]$new_device(x)
}

# A device object must belong to the backend an operation runs on: there is one
# active backend, and an array of another one cannot take part.
check_device_backend <- function(device, backend) {
  if (backend(device) != backend) {
    cli_abort(c(
      "{.arg device} belongs to the {.val {backend(device)}} backend, but the active backend is {.val {backend}}.",
      i = "Switch backends with {.fn with_backend} or {.fn local_backend}."
    ))
  }
  invisible(device)
}

#' Test whether an object is a device
#'
#' @param x An object to test.
#' @return `logical(1)`
#' @export
is_device <- function(x) {
  # TODO: device objects should share a common base class (like AnvlArray)
  # instead of checking each backend's class individually.
  inherits(x, c("PJRTDevice", "QuickrDevice"))
}
