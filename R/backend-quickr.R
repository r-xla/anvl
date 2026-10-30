#' @include backend-r.R
NULL

#' @title Quickr Device
#' @description
#' Device descriptor for the quickr backend. The only supported `type` is
#' `"cpu"`.
#'
#' @param x (`character(1)`)\cr
#'   Device type. Currently only supports `"cpu"`.
#' @return (`QuickrDevice`)
#' @seealso [`nv_device()`], [`AnvlBackendQuickr()`].
#' @export
quickr_device <- function(x = "cpu") {
  new_r_array_device(x, "QuickrDevice")
}

#' Quickr Backend
#'
#' Constructs the quickr backend, which stores array data as plain R arrays and
#' compiles jitted functions to R code via the \CRANpkg{quickr} package.
#'
#' Registered automatically under the name `"quickr"` when the package is
#' loaded; call [`local_backend("quickr")`][local_backend()] or
#' [`with_backend("quickr", ...)`][with_backend()] to use it. Requires the
#' quickr package to be installed.
#'
#' @section Data representation:
#' An [`AnvlArray`] with `backend = "quickr"` is, under the hood, a plain R
#' vector or array (`numeric`, `integer`, or `logical`) stored in the `$data`
#' field. [`as_array()`] returns the underlying vector/array directly without
#' copying, and [`nv_array()`] simply wraps an R vector/array. Data always lives
#' in R's memory and computation always runs on the CPU, so the only device is
#' [`quickr_device("cpu")`][quickr_device()]; every array still carries it in
#' `$device`.
#'
#' @section Status:
#' This backend is **experimental** and has a number of limitations:
#'
#' * Compilation (tracing + quickr lowering) is somewhat slow, so it is best
#'   suited to long-running or repeatedly-called functions where the one-time
#'   compilation cost is amortized. The [r backend][AnvlBackendR()] runs the
#'   same R function without compiling it.
#' * Only a subset of the primitives that the PJRT backend supports are currently
#'   lowered to quickr code. See
#'   `r roxy_article("primitives")` for an overview.
#' * Only CPU execution is supported.
#' * Only three data types are supported; see the section below.
#'
#' @section Supported data types:
#' `f64`, `i32` and `bool` -- the three R storage types.
#'
#' @return ([`AnvlBackend`])\cr
#'   With subclass `"AnvlBackendQuickr"`.
#' @seealso [`AnvlBackend()`], [`AnvlBackendPjrt()`], [`AnvlBackendR()`],
#'   [`local_backend()`], [`jit()`].
#' @export
AnvlBackendQuickr <- function() {
  backend <- r_array_backend("quickr", quickr_device, quickr_eager_compile)
  class(backend) <- c("AnvlBackendQuickr", class(backend))
  backend
}

register_backend("quickr", AnvlBackendQuickr())
