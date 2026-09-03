#' Create a backend
#'
#' @param new_data (`function`)\cr Constructs an AnvlArray from R data.
#' This should be a `structure()` with at least a `$data` field that contains the actual
#' underlying data (`PJRTBuffer` for `"pjrt"` backend, `array()` for `"quickr"` backend).
#' @param new_empty (`function`)\cr Constructs an AnvlArray of the given
#' `dtype` and `shape` with unspecified contents. Called by [`nv_empty()`].
#' @param dtype (`function`)\cr Extracts the dtype from an AnvlArray.
#' @param shape (`function`)\cr Extracts the shape from an AnvlArray.
#' @param as_array (`function(x, check)`)\cr Converts an AnvlArray to an R
#'   array. The `check` flag is forwarded from [`as_array()`]; backends may use
#'   it to abort when materialization would lose information (e.g. ui64 values
#'   wrapping through `bit64::integer64`). See [`pjrt::as_array.PJRTBuffer()`].
#' @param as_raw (`function`)\cr Converts an AnvlArray to raw bytes.
#' @param platform (`function`)\cr Returns the platform name (e.g. `"cpu"`).
#' @param device (`function`)\cr Returns the device object for an AnvlArray.
#' @param new_device (`function`)\cr Constructs a backend-specific device
#'   object from a device type string (e.g. `"cpu"`). Called by [`nv_device()`].
#' @param print_data (`function`)\cr Prints the array data with a footer.
#' @param jit (`function`)\cr Creates a JIT-compiled function implementation.
#' @param await_data (`function`)\cr Blocks until the array's underlying data
#'   is ready. Called by [`await()`] for `AnvlArray`s; a no-op for backends
#'   without async execution.
#' @return An `AnvlBackend` object.
#' @keywords internal
#' @export
AnvlBackend <- function(
  new_data,
  new_empty,
  dtype,
  shape,
  as_array,
  as_raw,
  platform,
  device,
  new_device,
  print_data,
  jit,
  await_data
) {
  structure(
    list(
      new_data = new_data,
      new_empty = new_empty,
      dtype = dtype,
      shape = shape,
      as_array = as_array,
      as_raw = as_raw,
      platform = platform,
      device = device,
      new_device = new_device,
      print_data = print_data,
      jit = jit,
      await_data = await_data
    ),
    class = "AnvlBackend"
  )
}

register_backend <- function(name, backend) {
  globals$backends[[name]] <- backend
}

# Compare two device objects for equality, returning FALSE when they are of
# different classes. Avoids R's "incompatible methods" warning when `==` is
# dispatched across device classes (e.g. PJRTDevice vs QuickrDevice).
eq_device <- function(x, y) {
  identical(class(x), class(y)) && isTRUE(x == y)
}

# Error when a traced graph contains arrays/devices from a backend other than
# `expected` (after accounting for `"plain"` constants, which are backend-agnostic).
# Catches both call-time inputs (via arg_devices) and closed-over constants,
# producing a clearer error than the downstream device-unification or
# codegen failures.
check_single_backend <- function(graph, arg_devices, expected) {
  const_backends <- vapply(
    graph$constants,
    function(const) if (is_concrete_tensor(const$aval)) backend(const$aval$data) else NA_character_,
    character(1)
  )
  arg_backends <- vapply(arg_devices, backend, character(1))
  found <- unique(c(const_backends, arg_backends))
  mismatches <- setdiff(found, c(expected, "plain", NA_character_))
  if (length(mismatches)) {
    cli_abort(c(
      "Cannot compile a {.val {expected}} program with inputs from other backends.",
      i = "Found arrays from backend{?s} {.val {mismatches}}.",
      i = "anvl does not support mixing backends in a single compiled program.",
      i = "Ensure all inputs and closed-over constants use the {.val {expected}} backend."
    ))
  }
}

PlainDeviceCpu <- function() {
  structure("cpu", class = "PlainDeviceCpu")
}

#' @export
format.PlainDeviceCpu <- function(x, ...) "PlainDeviceCpu"

#' @export
print.PlainDeviceCpu <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

globals$backends <- list()

# The plain backend is merely for capturing constants during jitting in a backend-agnostic way.
# Otherwise it is unused
register_backend(
  "plain",
  AnvlBackend(
    new_data = function(data, dtype, shape, device) {
      if (is.null(dtype)) {
        dtype <- default_dtype(data)
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
      structure(
        list(data = data, dtype = dtype, shape = shape, backend = "plain"),
        class = "AnvlArray"
      )
    },
    new_empty = function(dtype, shape, device) {
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
      data <- array(vector(storage_mode, prod(shape)), dim = shape)
      structure(
        list(data = data, dtype = dtype, shape = shape, backend = "plain"),
        class = "AnvlArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    as_array = function(x, check) x$data,
    as_raw = function(x, row_major) cli_abort("as_raw not supported for plain backend"),
    platform = function(x) "cpu",
    device = function(x) PlainDeviceCpu(),
    new_device = function(type) {
      cli_abort("{.val plain} backend does not support creating devices.")
    },
    print_data = function(x, footer) {
      print(x$data)
      cat(footer, "\n")
    },
    jit = function(f, static, cache_size, ...) {
      cli_abort("JIT compilation is not supported for the {.val plain} backend.")
    },
    await_data = function(x) invisible(NULL)
  )
)

#' Get the default backend
#'
#' Returns the current default backend from `getOption("anvl.default_backend", "pjrt")`.
#'
#' @return `character(1)` — the backend name (e.g. `"pjrt"`, `"quickr"`).
#' @seealso [local_backend()]
#' @export
default_backend <- function() {
  getOption("anvl.default_backend", "pjrt")
}

assert_backend <- function(backend) {
  assert_choice(backend, names(globals$backends))
}

#' Temporarily set the default backend
#'
#' Sets the `anvl.default_backend` option for the duration of the
#' calling scope. This affects `nv_array()`, `nv_scalar()`, and `jit()`.
#'
#' @param backend (`character(1)`)\cr
#'   Backend to use (`"pjrt"` or `"quickr"`).
#' @param envir The environment to scope the change to.
#' @return The previous value of the option (invisibly).
#' @export
local_backend <- function(backend, envir = parent.frame()) {
  backend <- assert_backend(backend)
  withr::local_options(anvl.default_backend = backend, .local_envir = envir)
}

#' Run code with a specific backend
#'
#' Sets the `anvl.default_backend` option for the duration of the
#' expression. This affects [`jit()`] and data construction (e.g. via [`nv_array`]).
#'
#' @param backend (`character(1)`)\cr
#'   Backend to use (`"pjrt"` or `"quickr"`).
#' @param code An expression to evaluate with the given backend.
#' @return The result of evaluating `code`.
#' @export
with_backend <- function(backend, code) {
  backend <- assert_backend(backend)
  withr::with_options(list(anvl.default_backend = backend), code)
}

#' Install what a backend needs to run
#'
#' A backend needs more than the packages anvl declares as dependencies: the
#' `"pjrt"` backend runs on PJRT plugins that are downloaded rather than shipped
#' with the package, and the `"quickr"` backend needs \CRANpkg{quickr}, which is only
#' suggested. This installs whichever of the two the given backend is missing.
#'
#' The PJRT plugins are downloaded on demand, but not silently: the first time a
#' plugin is needed, an interactive session asks for confirmation, while a
#' non-interactive session does not download at all. Call this to make the
#' download an explicit step instead, for instance in a `Dockerfile` layer of its
#' own or at the start of a script that later runs unattended. The `PJRT_INSTALL`
#' environment variable overrides the prompt: `"1"` always downloads without
#' asking, `"0"` never downloads.
#'
#' Which plugins you get -- and whether CUDA is available at all -- is decided
#' by the repository anvl was installed from, not by this call. See the
#' installation vignette: `vignette("installation", package = "anvl")`.
#'
#' @param backend (`character(1)`)\cr
#'   Backend to install for. Defaults to [default_backend()]. The `"plain"`
#'   backend has nothing to install and is not accepted.
#' @param ... Passed to the underlying installer: [pjrt::install_pjrt()] for
#'   `"pjrt"`, [utils::install.packages()] for `"quickr"`.
#' @return `NULL`, invisibly. Called for its side effect.
#' @export
install_anvl <- function(backend = default_backend(), ...) {
  backend <- assert_choice(backend, c("pjrt", "quickr"))
  switch(
    backend,
    pjrt = pjrt::install_pjrt(...),
    quickr = install.packages("quickr", ...)
  )
  invisible(NULL)
}
