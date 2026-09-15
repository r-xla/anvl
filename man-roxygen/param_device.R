#' @param device (`NULL` | `character(1)` | [device][nv_device])\cr
#'   The device the data lives on, given either as:
#'   * a *device string* naming the platform (e.g. `"cpu"`, `"cuda"`,
#'     `"cuda:<n>"`), which is resolved against the backend in use, or
#'   * a *device object* as returned by [`nv_device()`]: a
#'     [`PJRTDevice`][pjrt::pjrt_device] for the `"pjrt"` backend or a
#'     [`quickr_device`] for the `"quickr"` backend. Because a device object
#'     is backend-specific, it also determines the backend.
#'
#'   The default (`NULL`) uses [`default_device()`]: the CPU, or the platform
#'   named by the `PJRT_PLATFORM` environment variable on the `"pjrt"` backend.
