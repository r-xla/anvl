#' @include backend.R
NULL

# The backends that hold an array as an R array -- `"r"` and `"quickr"` -- share
# everything but their name, their device and what they do with the R function
# the quickr lowering emits for a jitted function: `"r"` runs it as it is,
# `"quickr"` compiles it first.

# The device of such a backend. Its arrays live in R's memory, so `"cpu"` is
# the only one; `class` names the backend the device belongs to (see
# `backend()`).
new_r_array_device <- function(x, class) {
  assert_choice(x, "cpu")
  structure(list(device = x), class = c(class, "RArrayDevice"))
}

#' @title R Device
#' @description
#' Device descriptor for the r backend. The only supported `type` is `"cpu"`.
#'
#' @param x (`character(1)`)\cr
#'   Device type. Currently only supports `"cpu"`.
#' @return (`RDevice`)
#' @seealso [`nv_device()`], [`AnvlBackendR()`].
#' @examples
#' r_device()
#' @export
r_device <- function(x = "cpu") {
  new_r_array_device(x, "RDevice")
}

#' @export
`==.RArrayDevice` <- function(e1, e2) {
  identical(class(e1), class(e2)) && e1$device == e2$device
}

#' @export
as.character.RArrayDevice <- function(x, ...) x$device

#' @export
format.RArrayDevice <- function(x, ...) paste0(class(x)[[1L]], "(", x$device, ")")

#' @export
print.RArrayDevice <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

# The methods of a backend that holds an array as an R array. `name` is the
# backend's, `new_device` builds its device from an identifier, and `compile`
# turns the R function the quickr lowering emits into the one a jitted function
# calls.
r_array_backend <- function(name, new_device, compile) {
  # There is only the one device, so a `device` argument just has to name it.
  check_device <- function(device) {
    if (!is.null(device)) {
      backend_device(device, name)
    }
  }
  new_array <- function(data, dtype, shape) {
    if (length(shape) >= 1L) {
      dim(data) <- shape
    }
    structure(
      list(
        data = data,
        dtype = dtype,
        shape = shape,
        # Stored rather than recomputed on demand: `$device` is part of what
        # identifies an array, and pjrt's dispatcher reads it off the leaf.
        device = new_device("cpu"),
        backend = name
      ),
      class = "AnvlArray"
    )
  }
  AnvlBackend(
    new_data = function(data, dtype, shape, device, row_major = FALSE) {
      check_device(device)
      if (!is_dtype(dtype)) {
        dtype <- as_dtype(dtype)
      }
      if (is.raw(data)) {
        return(new_array(r_array_from_raw(data, dtype, shape, row_major), dtype, shape))
      }
      # Errors for a data type an R array cannot hold, so an array is never
      # labelled with one it does not actually hold.
      new_array(quickr_dtype_info(dtype)$scalar_cast(data), dtype, shape %||% r_data_shape(data))
    },
    new_empty = function(dtype, shape, device) {
      check_device(device)
      if (!is_dtype(dtype)) {
        dtype <- as_dtype(dtype)
      }
      new_array(vector(quickr_dtype_to_r_ctor(dtype), prod(shape)), dtype, shape)
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    as_array = function(x, check) x$data,
    as_raw = function(x, row_major) {
      data <- x$data
      if (row_major && length(x$shape) >= 2L) {
        data <- aperm(data)
      }
      writeBin(as.vector(data), raw(), size = r_array_byte_width(x$dtype), endian = "little")
    },
    platform = function(x) "cpu",
    device = function(x) x$device,
    new_device = new_device,
    print_data = function(x, footer, ...) {
      print(x$data, ...)
      cat(footer, "\n")
    },
    jit = function(f, static, cache_size, unwrap = FALSE, device = NULL) {
      assert_flag(unwrap)
      jit_r_array_impl(f, static, cache_size, unwrap, device, name, compile)
    },
    await_data = function(x) invisible(NULL),
    # R has no single precision: an R double is a double.
    default_dtypes = list(float = "f64", int = "i32")
  )
}

# The bytes an element of an R array of data type `dtype` takes in a raw
# payload: those of the PJRT buffer of the same data type.
r_array_byte_width <- function(dtype) {
  switch(quickr_dtype_to_r_ctor(dtype), double = 8L, integer = 4L, logical = 1L)
}

# Decodes the little-endian bytes `data` -- in row-major order if `row_major` --
# into the R array of data type `dtype` and shape `shape`.
r_array_from_raw <- function(data, dtype, shape, row_major) {
  ctor <- quickr_dtype_to_r_ctor(dtype)
  width <- r_array_byte_width(dtype)
  n <- prod(shape)
  if (length(data) != n * width) {
    cli_abort(
      "A raw payload of {length(data)} bytes does not hold {n} element{?s} of data type {.val {as.character(dtype)}}."
    )
  }
  if (ctor == "logical") {
    values <- readBin(data, "integer", n = n, size = 1L, signed = FALSE) != 0L
  } else {
    values <- readBin(data, ctor, n = n, size = width, endian = "little")
  }
  if (row_major && length(shape) >= 2L) {
    values <- aperm(array(values, dim = rev(shape)))
  }
  values
}

# The native-dispatch compile callback: traces and lowers on a cache miss and
# hands the dispatcher the R closure to call. Reached only for inputs the
# dispatcher has already validated (see jit_pjrt_compile_cb).
jit_r_array_compile_cb <- function(f, static, unwrap, backend, compile) {
  function(info) {
    check_static_args(info$args, static)
    desc <- local_descriptor(default_dtypes = default_dtypes_from_key(info$context), backend = backend)
    graph <- trace_fn(f, desc = desc, args_flat = avals_from_dispatch(info), in_tree = info$in_tree)
    check_single_backend(graph, expected = backend)
    list(r_fun = graph_to_r_array_function(graph, backend, compile, unwrap = unwrap, flat = TRUE))
  }
}

jit_r_array_impl <- function(f, static, cache_size, unwrap, device, backend, compile) {
  if (!is.null(device)) {
    # There is one device, so there is nothing to place; this only rejects a
    # device of another backend.
    backend_device(device, backend)
  }
  # use pjrt's "closure" engine.
  dispatcher <- pjrt::dispatcher(
    cache_size,
    jit_r_array_compile_cb(f, static, unwrap, backend, compile),
    static = static,
    backend = backend,
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
    # Consulted only when a call has no array input to name a device. There is
    # one device today, but the dispatcher keys on whatever this returns, so a
    # second one would split the cache without further work here.
    default_device = function() default_device(backend),
    context = default_dtypes_context(backend)
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

#' R Backend
#'
#' Constructs the r backend, which stores array data as plain R arrays and runs
#' jitted functions as R code. It is primarily meant for testing: it needs
#' nothing beyond anvl itself -- no PJRT plugin and no compiler -- so it runs
#' wherever R does, and a package that uses anvl can run its tests on it where
#' no PJRT plugin can be downloaded, such as on CRAN. See
#' `r roxy_article("backends")` for how the three backends compare.
#'
#' Registered automatically under the name `"r"` when the package is loaded;
#' call [`local_backend("r")`][local_backend()] or
#' [`with_backend("r", ...)`][with_backend()] to use it.
#'
#' A jitted function is lowered to the same R function as on the
#' [quickr backend][AnvlBackendQuickr()], which is then run as it is instead of
#' being compiled by \CRANpkg{quickr}. Both backends therefore support the same
#' primitives and data types and compute the same results; the r backend has no
#' compilation cost, but runs the program at the speed of R code.
#'
#' @section Data representation:
#' An [`AnvlArray`] with `backend = "r"` is, under the hood, a plain R vector or
#' array (`numeric`, `integer`, or `logical`) stored in the `$data` field.
#' [`as_array()`] returns the underlying vector/array directly without copying,
#' and [`nv_array()`] simply wraps an R vector/array. The only device is
#' [`r_device("cpu")`][r_device()].
#'
#' @section Status:
#' This backend is **experimental**. Only a subset of the primitives that the
#' PJRT backend supports are currently lowered to R code. See
#' `r roxy_article("primitives")` for an overview (the column of the quickr
#' backend).
#'
#' @section Supported data types:
#' `f64`, `i32` and `bool` -- the three R storage types.
#'
#' @return ([`AnvlBackend`])\cr
#'   With subclass `"AnvlBackendR"`.
#' @seealso [`AnvlBackend()`], [`AnvlBackendPjrt()`], [`AnvlBackendQuickr()`],
#'   [`local_backend()`], [`jit()`].
#' @export
AnvlBackendR <- function() {
  backend <- r_array_backend("r", r_device, plain_r_function)
  class(backend) <- c("AnvlBackendR", class(backend))
  backend
}

register_backend("r", AnvlBackendR())
