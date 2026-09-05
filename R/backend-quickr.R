#' @include backend.R
NULL

#' @title Quickr device
#' @description
#' Device descriptor for the quickr backend. The only supported `type` is
#' `"cpu"`.
#'
#' @param x (`character(1)`)\cr
#'   Device type. Currently only supports `"cpu"`.
#' @return A `QuickrDevice` object.
#' @seealso [`nv_device()`], [`AnvlBackendQuickr()`].
#' @export
quickr_device <- function(x = "cpu") {
  assert_choice(x, c("cpu"))
  structure(list(device = x), class = "QuickrDevice")
}

#' @export
`==.QuickrDevice` <- function(e1, e2) {
  e1$device == e2$device
}

#' @export
as.character.QuickrDevice <- function(x, ...) x$device

#' @export
format.QuickrDevice <- function(x, ...) paste0("QuickrDevice(", x$device, ")")

#' @export
print.QuickrDevice <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

# The native-dispatch compile callback: traces and quickr-compiles on a cache
# miss and hands the dispatcher the compiled R closure. Reached only for inputs
# the dispatcher has already validated (see jit_pjrt_compile_cb).
jit_quickr_compile_cb <- function(f, static, unwrap, device) {
  function(info) {
    check_static_args(info$args, static)
    if (is_device_arg(device)) {
      # quickr has one device, but the one asked for still has to be its own.
      device_from_arg(info$args[[device$argname]], "quickr")
    }
    compiled <- compile_quickr(
      f,
      args_flat = avals_from_dispatch(info),
      in_tree = info$in_tree,
      arg_devices = dispatch_arg_devices(info),
      unwrap = unwrap,
      flat = TRUE,
      default_dtypes = default_dtypes_from_key(info$context)
    )
    list(r_fun = compiled$fun)
  }
}

jit_quickr_impl <- function(f, static, cache_size, unwrap, device) {
  if (!is.null(device) && !is_device_arg(device)) {
    # quickr has one device, so there is nothing to place; this only rejects a
    # device of another backend.
    backend_device(device, "quickr")
  }
  # use pjrt's "closure" engine for quickr.
  dispatcher <- pjrt::dispatcher(
    cache_size,
    jit_quickr_compile_cb(f, static, unwrap, device),
    static = static,
    backend = "quickr",
    # The dispatcher reads a non-pjrt leaf's metadata through this, via the
    # backend's accessor generics -- so an AnvlArray need only carry $data, per
    # the AnvlBackend contract, not store dtype/shape/device as fields.
    extractor = function(leaf) {
      list(
        aval = list(dtype = dtype(leaf), shape = shape(leaf)),
        device = device(leaf),
        backend = backend(leaf)
      )
    },
    # Consulted only when a call has no array input to name a device. quickr has
    # one device today, but the dispatcher keys on whatever this returns, so a
    # second one would split the cache without further work here.
    default_device = function() default_device("quickr"),
    # See jit_pjrt_impl().
    context = default_dtypes_context("quickr")
  )
  dispatch <- pjrt::dispatch

  # The dispatcher validates the inputs itself and errors on anything the
  # compiled closure cannot take, so there is no fallback. The compiled
  # closure's return value is the call's result, verbatim.
  run <- function(args) {
    dispatch(dispatcher, args)
  }

  fn <- function() {
    # calling a jitted function within another jitted function --> re-trace the original closure
    if (currently_tracing()) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    run(args)
  }
  attr(fn, "jit_run_args") <- run
  fn
}

compile_quickr <- function(
  f,
  args_flat,
  in_tree,
  arg_devices = list(),
  unwrap = FALSE,
  flat = FALSE,
  default_dtypes = NULL
) {
  desc <- local_descriptor(default_dtypes = default_dtypes)
  graph <- trace_fn(f, desc = desc, args_flat = args_flat, in_tree = in_tree, mode = "toplevel")
  check_single_backend(graph, arg_devices = arg_devices, expected = "quickr")
  list(fun = graph_to_quickr_function(graph, unwrap = unwrap, flat = flat))
}

#' Quickr backend
#'
#' Constructs the quickr backend, which stores array data as plain R arrays and
#' compiles jitted functions to R code via the \CRANpkg{quickr} package.
#'
#' To use it, the `"quickr"` package needs to be installed.
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
#' `$device`, as arrays of every backend do.
#'
#' @section Status:
#' This backend is **experimental** and has a number of limitations:
#'
#' * Compilation (tracing + quickr lowering) is somewhat slow, so it is best
#'   suited to long-running or repeatedly-called functions where the one-time
#'   compilation cost is amortized.
#' * Only a subset of the primitives that the PJRT backend supports are currently
#'   lowered to quickr code. See `vignette("primitives")` for an overview.
#' * Only the data types `f64`, `i32`, and `bool` are supported. Accordingly,
#'   an R double commits to `f64` on this backend (see [`default_dtypes()`]),
#'   and any other float -- `f32` included, since quickr has no single
#'   precision -- is an error rather than a double wearing the wrong label.
#' * Only CPU execution is supported.
#'
#' @section Quickr JIT arguments:
#'
#' * `unwrap` (`logical(1)`, default `FALSE`): if `TRUE`, the compiled function
#'   returns plain R arrays instead of [`AnvlArray`]s. Useful when the jitted
#'   function's output is consumed by non-anvl R code and the extra wrapping
#'   would only get stripped again.
#'
#' @return An [`AnvlBackend`] object with subclass `"AnvlBackendQuickr"`.
#' @seealso [`AnvlBackend()`], [`AnvlBackendPjrt()`], [`local_backend()`], [`jit()`].
#' @export
AnvlBackendQuickr <- function() {
  backend <- AnvlBackend(
    new_data = function(data, dtype, shape, device) {
      if (!is.null(device)) {
        if (is.character(device) && (device != "quickr")) {
          cli_abort("Unsupported device {.val {device}} for 'quickr' backend")
        } else if (!inherits(device, "QuickrDevice")) {
          cli_abort("Invalid device of class {.cls {class(device)}} for 'quickr' backend")
        }
      }
      if (!is_dtype(dtype)) {
        dtype <- as_dtype(dtype)
      }
      if (is.null(shape)) {
        shape <- if (!is.null(dim(data))) {
          as.integer(dim(data))
        } else if (length(data) == 1L) {
          1L
        } else {
          as.integer(length(data))
        }
      }
      # Errors for a data type quickr cannot represent, so an array is never
      # labelled with one it does not actually hold.
      data <- quickr_dtype_info(dtype)$scalar_cast(data)
      if (length(shape) >= 1L) {
        dim(data) <- shape
      }
      structure(
        list(
          data = data,
          dtype = dtype,
          shape = shape,
          # quickr is CPU-only, so every accepted `device` is this one. It is
          # stored rather than recomputed on demand: `$device` is part of what
          # identifies an array, and pjrt's dispatcher reads it off the leaf.
          device = quickr_device("cpu"),
          backend = "quickr"
        ),
        class = "AnvlArray"
      )
    },
    new_empty = function(dtype, shape, device) {
      if (!is.null(device)) {
        if (is.character(device) && (device != "quickr")) {
          cli_abort("Unsupported device {.val {device}} for 'quickr' backend")
        } else if (!inherits(device, "QuickrDevice")) {
          cli_abort("Invalid device of class {.cls {class(device)}} for 'quickr' backend")
        }
      }
      if (!is_dtype(dtype)) {
        dtype <- as_dtype(dtype)
      }
      storage_mode <- switch(
        quickr_dtype_to_r_ctor(dtype),
        "double" = "double",
        "integer" = "integer",
        "logical" = "logical",
        "double"
      )
      data <- vector(storage_mode, prod(shape))
      if (length(shape) >= 1L) {
        dim(data) <- shape
      }
      structure(
        list(
          data = data,
          dtype = dtype,
          shape = shape,
          device = quickr_device("cpu"),
          backend = "quickr"
        ),
        class = "AnvlArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    as_array = function(x, check) x$data,
    as_raw = function(x, row_major) as.raw(x$data),
    platform = function(x) "cpu",
    device = function(x) x$device,
    new_device = function(x) quickr_device(x),
    print_data = function(x, footer) {
      print(x$data)
      cat(footer, "\n")
    },
    jit = function(f, static, cache_size, unwrap = FALSE, device = NULL) {
      assert_flag(unwrap)
      jit_quickr_impl(f, static, cache_size, unwrap, device)
    },
    await_data = function(x) invisible(NULL),
    # quickr has no single precision: an R double is a double.
    default_dtypes = list(float = "f64", int = "i32")
  )
  class(backend) <- c("AnvlBackendQuickr", class(backend))
  backend
}

register_backend("quickr", AnvlBackendQuickr())
