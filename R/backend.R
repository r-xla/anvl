#' Create a backend
#'
#' @param new_data (`function`)\cr Constructs an AnvlArray from R data.
#' This should be a `structure()` with at least a `$data` field that contains the actual
#' underlying data (`PJRTBuffer` for `"pjrt"` backend, `array()` for `"quickr"` backend).
#' @param new_empty (`function`)\cr Constructs an AnvlArray of the given
#' `dtype` and `shape` with unspecified contents. Called by [`nv_empty()`].
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
  new_empty,
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
      new_empty = new_empty,
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

# Every backend anvl ships, as a constructor per name. Which of them are
# actually registered is decided once at load time from the `anvl.backends`
# option; see `activate_backends()`. This is a function (not a list built at
# source time) so it does not depend on the collate order.
backend_constructors <- function() {
  list(pjrt = AnvlBackendPjrt, quickr = AnvlBackendQuickr)
}

# Validate the `anvl.backends` option and derive the initial default backend
# from it. A plain function of its input so it is testable without reloading
# the namespace.
resolve_backends <- function(backends) {
  known <- names(backend_constructors())
  if (!is.character(backends) || !length(backends) || anyNA(backends)) {
    cli_abort(c(
      "Option {.code anvl.backends} must be a non-empty character vector of backend names.",
      i = "Known backends: {.val {known}}."
    ))
  }
  backends <- unique(backends)
  unknown <- setdiff(backends, known)
  if (length(unknown)) {
    cli_abort(c(
      "Option {.code anvl.backends} contains unknown backend{?s} {.val {unknown}}.",
      i = "Known backends: {.val {known}}."
    ))
  }
  list(
    active = backends,
    # With more than one active backend there is no single sensible default, so
    # the backend is picked from the call's inputs instead.
    default = if (length(backends) > 1L) "auto" else backends
  )
}

# Install the backend registry for the given set of active backends. Called
# from `.onLoad()` with the value of the `anvl.backends` option.
activate_backends <- function(backends) {
  resolved <- resolve_backends(backends)
  globals$active_backends <- resolved$active
  globals$default_backend <- resolved$default
  globals$backends <- c(
    list(plain = AnvlBackendPlain()),
    lapply(backend_constructors()[resolved$active], function(ctor) ctor())
  )
  invisible(resolved)
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

# The plain backend is merely for capturing constants during jitting in a backend-agnostic way.
# Otherwise it is unused. It is always registered, independently of `anvl.backends`.
AnvlBackendPlain <- function() {
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
    new_empty = function(dtype, shape, device, ambiguous) {
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
    jit = function(f, static, cache_size, ...) {
      cli_abort("JIT compilation is not supported for the {.val plain} backend.")
    },
    await_data = function(x) invisible(NULL)
  )
}

#' Get the default backend
#'
#' Returns the current default backend, i.e. the value of the
#' `anvl.default_backend` option. Its initial value is derived from the active
#' backends (see [`anvl-package`]): the single active backend, or `"auto"` when
#' several backends are active.
#'
#' @return `character(1)` — the backend name (e.g. `"pjrt"`, `"quickr"`, `"auto"`).
#' @seealso [local_backend()], [`with_backend()`]
#' @export
default_backend <- function() {
  getOption("anvl.default_backend", globals$default_backend)
}

# `"auto"` defers the backend choice to the call's inputs. A constructor that
# has neither an array nor a device to infer from falls back to the first
# active backend.
resolve_eager_backend <- function(backend) {
  if (identical(backend, "auto")) globals$active_backends[[1L]] else backend
}

# Ensure `backend` is one anvl can actually use, with a hint pointing at
# `anvl.backends` when it is a known but inactive backend.
assert_backend <- function(backend, auto = FALSE) {
  assert_string(backend)
  # `names(globals$backends)` is the active backends plus the internal "plain"
  # one, which constructors accept for trace-time constants.
  if (backend %in% c(names(globals$backends), if (auto) "auto")) {
    return(invisible(backend))
  }
  if (backend %in% names(backend_constructors())) {
    hint <- deparse1(union(globals$active_backends, backend))
    cli_abort(c(
      "Backend {.val {backend}} is not active.",
      i = "Active backend{?s}: {.val {globals$active_backends}}.",
      i = "Set {.code options(anvl.backends = {hint})} before loading anvl to activate it."
    ))
  }
  cli_abort(c(
    "Unknown backend {.val {backend}}.",
    i = "Active backend{?s}: {.val {globals$active_backends}}."
  ))
}

#' Temporarily set the default backend
#'
#' Sets the `anvl.default_backend` option for the duration of the
#' calling scope. This affects `nv_array()`, `nv_scalar()`, and `jit()`.
#'
#' @param backend (`character(1)`)\cr
#'   Backend to use; one of the active backends (see [`anvl-package`]) or
#'   `"auto"`.
#' @param envir The environment to scope the change to.
#' @return The previous value of the option (invisibly).
#' @export
local_backend <- function(backend, envir = parent.frame()) {
  assert_backend(backend, auto = TRUE)
  withr::local_options(anvl.default_backend = backend, .local_envir = envir)
}

#' Run code with a specific backend
#'
#' Sets the `anvl.default_backend` option for the duration of the
#' expression. This affects [`jit()`] and data construction (e.g. via [`nv_array`]).
#'
#' @param backend (`character(1)`)\cr
#'   Backend to use; one of the active backends (see [`anvl-package`]) or
#'   `"auto"`.
#' @param code An expression to evaluate with the given backend.
#' @return The result of evaluating `code`.
#' @export
with_backend <- function(backend, code) {
  assert_backend(backend, auto = TRUE)
  withr::with_options(list(anvl.default_backend = backend), code)
}
