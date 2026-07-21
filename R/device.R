#' @title Get the default device
#' @description
#' Returns a device object for the default backend and platform.
#' For the `"pjrt"` backend, the platform is determined by the `PJRT_PLATFORM`
#' environment variable (defaulting to `"cpu"`). Other backends (e.g. `"quickr"`)
#' only support CPU. The backend defaults to [`default_backend()`].
#' @param backend (`NULL` | `character(1)`)\cr
#'   Backend. Defaults to [`default_backend()`] when `NULL`.
#' @return A backend-specific device object.
#' @seealso [`nv_device()`], [`default_backend()`]
#' @export
default_device <- function(backend = NULL) {
  backend <- resolve_eager_backend(backend %||% default_backend())
  platform <- if (backend == "pjrt") Sys.getenv("PJRT_PLATFORM", "cpu") else "cpu"
  nv_device(platform, backend)
}

#' @title Create a Device
#' @description
#' Constructs a backend-specific device object.
#'
#' A device identifies a compute resources, such as CPU, or a specific GPU.
#' It is relevant for data allocation (e.g. via [nv_array()]) but also compilation ([jit]).
#'
#' @param x (`character(1)` | device object)\cr
#'   Identifier for the device (e.g. `"cpu"`, `"cuda"`, `"cuda:<n>"`),
#'   or an existing device object (returned as-is).
#' @param backend (`NULL` | `character(1)`)\cr
#'   The backend for which to create the device.
#'   Defaults to [`default_backend()`] when `NULL`.
#' @return A backend-specific device object (e.g. `PJRTDevice` for `"pjrt"`,
#'   [`quickr_device`] for `"quickr"`).
#' @seealso [`backend()`], [`AnvlBackend()`].
#' @examplesIf pjrt::plugins_downloaded()
#' # Create CPU device for pjrt backend:
#' nv_device("cpu", "pjrt")
#' # Create CPU device for quickr backend:
#' nv_device("cpu", "quickr")
#' # Pass through an existing device:
#' dev <- nv_device("cpu")
#' identical(nv_device(dev), dev)
#' @export
nv_device <- function(x, backend = NULL) {
  if (is_device(x)) {
    if (!is.null(backend) && backend(x) != backend) {
      cli_abort(
        "{.arg x} has backend {.val {backend(x)}}, but {.arg backend} is {.val {backend}}."
      )
    }
    return(x)
  }
  backend <- resolve_eager_backend(backend %||% default_backend())
  assert_backend(backend)
  globals$backends[[backend]]$new_device(x)
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
