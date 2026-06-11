#' Create a backend
#'
#' @param new_data (`function`)\cr Constructs an AnvlArray from R data.
#' This should be a `structure()` with at least a `$data` field that contains the actual
#' underlying data (`PJRTBuffer` for `"xla"` backend, `array()` for `"quickr"` backend).
#' @param dtype (`function`)\cr Extracts the dtype from an AnvlArray.
#' @param shape (`function`)\cr Extracts the shape from an AnvlArray.
#' @param ambiguous (`function`)\cr Extracts the ambiguous flag from an AnvlArray.
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
  dtype,
  shape,
  ambiguous,
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
      dtype = dtype,
      shape = shape,
      ambiguous = ambiguous,
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
    new_data = function(data, dtype, shape, device, ambiguous) {
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
        list(data = data, dtype = dtype, shape = shape, ambiguous = ambiguous, backend = "plain"),
        class = "AnvlArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    ambiguous = function(x) x$ambiguous,
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
    jit = function(f, static, cache, ...) {
      cli_abort("JIT compilation is not supported for the {.val plain} backend.")
    },
    await_data = function(x) invisible(NULL)
  )
)

#' Get the default backend
#'
#' Returns the current default backend from `getOption("anvl.default_backend", "xla")`.
#'
#' @return `character(1)` — the backend name (e.g. `"xla"`, `"quickr"`).
#' @seealso [local_backend()]
#' @export
default_backend <- function() {
  getOption("anvl.default_backend", "xla")
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
#'   Backend to use (`"xla"` or `"quickr"`).
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
#'   Backend to use (`"xla"` or `"quickr"`).
#' @param code An expression to evaluate with the given backend.
#' @return The result of evaluating `code`.
#' @export
with_backend <- function(backend, code) {
  backend <- assert_backend(backend)
  withr::with_options(list(anvl.default_backend = backend), code)
}
