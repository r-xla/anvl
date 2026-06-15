#' @include backend.R
NULL

#' @title Quickr device
#' @description
#' Device descriptor for the quickr backend. The only supported `type` is
#' `"cpu"`.
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

jit_quickr_impl <- function(f, static, cache, unwrap) {
  function() {
    # calling a jitted function within another jitted function --> re-trace the original closure
    if (currently_tracing()) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    prep <- jit_prepare_call(match.call(), parent.frame(), static, backend = "quickr")
    avals_in <- to_avals(prep$args_flat, prep$is_static_flat)

    args_flat_nv <- prep$args_flat[!prep$is_static_flat & vapply(prep$args_flat, is_anvl_array, logical(1))]
    arg_devices <- lapply(args_flat_nv, tengen::device)

    cache_key <- list(prep$in_tree, avals_in)
    r_args_flat <- lapply(prep$args_flat, function(a) {
      if (is_anvl_array(a)) as_array(a) else a
    })
    cache_hit <- cache$get(cache_key)
    if (!is.null(cache_hit)) {
      return(cache_hit(r_args_flat))
    }

    compiled <- compile_quickr(
      f,
      args_flat = avals_in,
      in_tree = prep$in_tree,
      arg_devices = arg_devices,
      unwrap = unwrap,
      flat = TRUE
    )
    cache$set(cache_key, compiled$fun)
    compiled$fun(r_args_flat)
  }
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
#' compiles jitted functions to R code via the
#' [quickr](https://CRAN.R-project.org/package=quickr) package.
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
#' copying, and [`nv_array()`] simply wraps an R vector/array. As a
#' consequence, there is no separate notion of a device: data always lives in
#' R's memory and computation always runs on the CPU.
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
        list(data = data, dtype = dtype, shape = shape, ambiguous = ambiguous, backend = "quickr"),
        class = "AnvlArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    ambiguous = function(x) x$ambiguous,
    as_array = function(x, check) x$data,
    as_raw = function(x, row_major) as.raw(x$data),
    platform = function(x) "cpu",
    device = function(x) quickr_device("cpu"),
    new_device = function(x) quickr_device(x),
    print_data = function(x, footer) {
      print(x$data)
      cat(footer, "\n")
    },
    jit = function(f, static, cache, unwrap = FALSE, device = NULL) {
      assert_flag(unwrap)
      jit_quickr_impl(f, static, cache, unwrap)
    },
    await_data = function(x) invisible(NULL)
  )
  class(backend) <- c("AnvlBackendQuickr", class(backend))
  backend
}

register_backend("quickr", AnvlBackendQuickr())
