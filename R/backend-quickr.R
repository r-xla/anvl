#' @include backend.R
NULL

#' @title Quickr device
#' @description
#' Device descriptor for the quickr backend. The only supported `type` is
#' `"cpu"`.
#'
#' Each call returns a fresh object; the quickr backend does not intern its
#' devices. pjrt's dispatcher canonicalizes devices with `identical()` as a
#' fallback to object identity, so equal-but-distinct `QuickrDevice` objects
#' still collapse to one device -- and quickr has a single device, so interning
#' would buy nothing.
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
# the dispatcher has already validated (see jit_xla_compile_cb).
jit_quickr_compile_cb <- function(f, unwrap) {
  function(info) {
    compiled <- compile_quickr(
      f,
      args_flat = avals_from_dispatch(info),
      in_tree = info$in_tree,
      arg_devices = dispatch_arg_devices(info),
      unwrap = unwrap,
      flat = TRUE
    )
    list(r_fun = compiled$fun)
  }
}

jit_quickr_impl <- function(f, static, cache_size, unwrap) {
  # pjrt's native dispatcher owns the cache; entries hold the quickr-compiled
  # R closure -- the "quickr" backend (anything but "xla") selects pjrt's
  # closure engine, which is called on the flat leaves: R-array-backed
  # AnvlArrays contribute their $data, everything else passes through (the
  # closure's wrapper checks and drops the static slots).
  dispatcher <- pjrt::dispatcher(
    cache_size,
    jit_quickr_compile_cb(f, unwrap),
    static = static,
    backend = "quickr",
    # The dispatcher reads a non-xla leaf's metadata through this, via the
    # backend's accessor generics -- so an AnvlArray need only carry $data, per
    # the AnvlBackend contract, not store dtype/shape/device as fields.
    extractor = function(leaf) {
      list(
        aval = list(dtype = dtype(leaf), shape = shape(leaf), ambiguous = ambiguous(leaf)),
        device = device(leaf),
        backend = backend(leaf)
      )
    },
    # Consulted only when a call has no array input to name a device. quickr has
    # one device today, but the dispatcher keys on whatever this returns, so a
    # second one would split the cache without further work here.
    default_device = function() default_device("quickr")
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

compile_quickr <- function(f, args_flat, in_tree, arg_devices = list(), unwrap = FALSE, flat = FALSE) {
  desc <- local_descriptor()
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
#' * Only a subset of the primitives that the XLA backend supports are currently
#'   lowered to quickr code. See `vignette("primitives")` for an overview.
#' * Only the data types `f64`, `i32`, and `bool` are supported.
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
#' @seealso [`AnvlBackend()`], [`AnvlBackendXla()`], [`local_backend()`], [`jit()`].
#' @export
AnvlBackendQuickr <- function() {
  backend <- AnvlBackend(
    new_data = function(data, dtype, shape, device, ambiguous) {
      if (!is.null(device)) {
        if (is.character(device) && (device != "quickr")) {
          cli_abort("Unsupported device {.val {device}} for 'quickr' backend")
        } else if (!inherits(device, "QuickrDevice")) {
          cli_abort("Invalid device of class {.cls {class(device)}} for 'quickr' backend")
        }
      }
      if (is.null(dtype)) {
        dtype <- if (is.double(data)) FloatType(64) else default_dtype(data)
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
      dtype_chr <- as.character(dtype)
      data <- switch(
        substr(dtype_chr, 1, 1),
        "f" = as.double(data),
        "i" = ,
        "u" = as.integer(data),
        "b" = as.logical(data),
        as.double(data)
      )
      if (length(shape) >= 1L) {
        dim(data) <- shape
      }
      structure(
        list(
          data = data,
          dtype = dtype,
          shape = shape,
          ambiguous = ambiguous,
          # quickr is CPU-only, so every accepted `device` is this one. It is
          # stored rather than recomputed on demand: `$device` is part of what
          # identifies an array, and pjrt's dispatcher reads it off the leaf.
          device = quickr_device("cpu"),
          backend = "quickr"
        ),
        class = "AnvlArray"
      )
    },
    new_empty = function(dtype, shape, device, ambiguous) {
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
        substr(as.character(dtype), 1L, 1L),
        "f" = "double",
        "i" = ,
        "u" = "integer",
        "b" = "logical",
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
          ambiguous = ambiguous,
          device = quickr_device("cpu"),
          backend = "quickr"
        ),
        class = "AnvlArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    ambiguous = function(x) x$ambiguous,
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
      jit_quickr_impl(f, static, cache_size, unwrap)
    },
    await_data = function(x) invisible(NULL)
  )
  class(backend) <- c("AnvlBackendQuickr", class(backend))
  backend
}

register_backend("quickr", AnvlBackendQuickr())
