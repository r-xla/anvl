#' @title Get the default device
#' @description
#' Returns the default device of the active backend: the device the
#' `anvl.default_device` option names (see [`local_default_device()`]), else
#' the one the `ANVL_DEFAULT_DEVICE` environment variable names (e.g.
#' `ANVL_DEFAULT_DEVICE=cuda`, read once when anvl is loaded), else the first
#' CPU device.
#' @param backend (`NULL` | `character(1)`)\cr
#'   Backend. Defaults to [`active_backend()`] when `NULL`.
#' @return (device object)\cr
#'   Backend-specific.
#' @seealso [`nv_device()`], [`active_backend()`], [`local_default_device()`]
#' @export
default_device <- function(backend = NULL) {
  backend <- backend %||% active_backend()
  device <- getOption("anvl.default_device") %||% globals[["ENV_DEFAULT_DEVICE"]] %||% "cpu"
  backend_device(device, backend)
}

#' @title Temporarily Set the Default Device
#' @description
#' Sets the `anvl.default_device` option, which [`default_device()`] returns in
#' place of the first CPU device: `local_default_device()` for the
#' calling scope, `with_default_device()` for one expression.
#'
#' This is what a call that names no device allocates on, and what a jitted
#' function whose graph pins no device of its own compiles for.
#' @param device (`character(1)` | device object)\cr
#'   The device to make the default, e.g. `"cpu:1"`. A string is looked up on
#'   whichever backend asks for the default, so an identifier only one backend
#'   knows (`"cpu:1"` is beyond quickr's single device) makes the default an
#'   error on the others.
#' @param code (`any`)\cr
#'   Expression to evaluate with the default device set.
#' @param envir (`environment`)\cr
#'   Scope the option is reset at the end of. Defaults to the caller.
#' @return `local_default_device()` returns the previous option value
#'   invisibly, `with_default_device()` the value of `code`.
#' @seealso [`default_device()`], [`local_backend()`]
#' @examplesIf pjrt::plugins_downloaded()
#' with_default_device("cpu:0", device(nv_array(1:3)))
#' @export
local_default_device <- function(device, envir = parent.frame()) {
  withr::local_options(
    list(anvl.default_device = check_default_device(device)),
    .local_envir = envir
  )
}

#' @rdname local_default_device
#' @export
with_default_device <- function(device, code) {
  withr::with_options(list(anvl.default_device = check_default_device(device)), code)
}

# The option holds an identifier rather than a device object, because every
# backend resolves it for itself. `NULL` clears it.
check_default_device <- function(device) {
  if (is.null(device) || is_device(device)) {
    return(device)
  }
  assert_string(device, .var.name = "device")
  device
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
#' @return (device object)\cr
#'   Backend-specific (e.g. `PJRTDevice` for `"pjrt"`,
#'   [`quickr_device`] for `"quickr"`).
#' @seealso [`backend()`], [`AnvlBackend()`], [`active_backend()`].
#' @examplesIf pjrt::plugins_downloaded()
#' # create CPU device for the active backend
#' nv_device("cpu")
#' # create CPU device for the quickr backend:
#' with_backend("quickr", nv_device("cpu"))
#' # pass through an existing device:
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
#' @return (`logical(1)`)\cr
#'   Whether `x` is a device of one of the backends.
#' @export
is_device <- function(x) {
  # TODO: device objects should share a common base class (like AnvlArray)
  # instead of checking each backend's class individually.
  inherits(x, c("PJRTDevice", "QuickrDevice"))
}

# The device a value pins an operation to, or `NULL` when it pins none. Only a
# concrete array does: a `"plain"` constant is backend-agnostic -- its
# `PlainDeviceCpu()` stands in for a device rather than being one -- and a
# traced value is placed by `jit()` when the graph is compiled.
placement_device <- function(x) {
  if (is_anvl_array(x) && backend(x) != "plain") {
    device(x)
  }
}
