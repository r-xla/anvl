#' @title Get the default device
#' @description
#' Returns the default device of the backend in force and the default platform.
#' For the `"pjrt"` backend, the platform is determined by the `PJRT_PLATFORM`
#' environment variable (defaulting to `"cpu"`). Other backends (e.g. `"quickr"`)
#' only support CPU.
#' @param backend (`NULL` | `character(1)`)\cr
#'   Backend. Defaults to [`default_backend()`] when `NULL`.
#' @return A backend-specific device object.
#' @seealso [`nv_device()`], [`default_backend()`]
#' @export
default_device <- function(backend = NULL) {
  backend <- backend %||% default_backend()
  platform <- if (backend == "pjrt") Sys.getenv("PJRT_PLATFORM", "cpu") else "cpu"
  backend_device(platform, backend)
}

#' @title Create a Device
#' @description
#' Constructs a backend-specific device object.
#'
#' A device identifies a compute resources, such as CPU, or a specific GPU.
#' It is relevant for data allocation (e.g. via [nv_array()]) but also compilation ([jit]).
#' A device belongs to the backend in force ([`default_backend()`]); a device
#' object of another backend is an error.
#'
#' @param x (`character(1)` | device object)\cr
#'   Identifier for the device (e.g. `"cpu"`, `"cuda"`, `"cuda:<n>"`),
#'   or an existing device object of the backend in force (returned as-is).
#' @return A backend-specific device object (e.g. `PJRTDevice` for `"pjrt"`,
#'   [`quickr_device`] for `"quickr"`).
#' @seealso [`backend()`], [`AnvlBackend()`], [`default_backend()`].
#' @examplesIf pjrt::plugins_downloaded()
#' # Create CPU device for the pjrt backend:
#' nv_device("cpu")
#' # Create CPU device for the quickr backend:
#' with_backend("quickr", nv_device("cpu"))
#' # Pass through an existing device:
#' dev <- nv_device("cpu")
#' identical(nv_device(dev), dev)
#' @export
nv_device <- function(x) {
  backend_device(x, default_backend())
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

# The value of a `device_arg()` argument at call time, as a device of `backend`:
# NULL when the caller gave none, so the device is inferred.
device_from_arg <- function(x, backend) {
  if (is.null(x)) {
    return(NULL)
  }
  backend_device(x, backend)
}

# A device object must belong to the backend an operation runs on: there is one
# backend in force, and an array of another one cannot take part.
check_device_backend <- function(device, backend) {
  if (backend(device) != backend) {
    cli_abort(c(
      "{.arg device} belongs to the {.val {backend(device)}} backend, but the backend in force is {.val {backend}}.",
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
