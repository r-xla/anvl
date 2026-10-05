#' @include default-dtypes.R
NULL

#' Create a Backend
#'
#' @param new_data (`function(data, dtype, shape, device, row_major = FALSE)`)\cr
#'   Constructs an AnvlArray from R data. Must return
#'   `structure(list(data = , backend = <name>, ...), class = "AnvlArray")`, where
#'   `data` holds the underlying data (a `PJRTBuffer` for the `"pjrt"` backend, an
#'   R `array()` for the `"quickr"` backend) and `backend` is the name the
#'   backend is registered under. `row_major` gives the element order of raw
#'   byte payloads; a backend that does not support raw `data` should abort on
#'   it.
#' @param new_empty (`function(dtype, shape, device)`)\cr
#'   Constructs an AnvlArray of the given `dtype` and `shape` with unspecified
#'   contents. Called by [`nv_empty()`].
#' @param dtype (`function(x)`)\cr Extracts the dtype from an AnvlArray.
#' @param shape (`function(x)`)\cr Extracts the shape from an AnvlArray.
#' @param as_array (`function(x, check)`)\cr Converts an AnvlArray to an R
#'   array. The `check` level is forwarded from [`as_array()`]; backends may use
#'   it to abort when materialization would lose information (e.g. ui64 values
#'   wrapping through `bit64::integer64`). See [`pjrt::as_array.PJRTBuffer()`].
#' @param as_raw (`function(x, row_major)`)\cr Converts an AnvlArray to raw
#'   bytes.
#' @param platform (`function(x)`)\cr Returns the platform name (e.g. `"cpu"`).
#' @param device (`function(x)`)\cr Returns the device object for an AnvlArray.
#' @param new_device (`function(x)`)\cr Constructs a backend-specific device
#'   object from a device identifier (e.g. `"cpu"` or `"cuda:1"`). Called by
#'   [`nv_device()`].
#' @param platform_devices (`function(device)`)\cr Returns a `list` of all
#'   devices of the platform `device` belongs to. Called by [`device_map()`].
#' @param copy_to_device (`function(x, device)`)\cr Returns a copy of the
#'   AnvlArray `x` on `device`, without waiting for the copy to finish where
#'   the backend runs asynchronously. Called by [`device_map()`].
#' @param print_data (`function(x, footer, ...)`)\cr Prints the array data with a
#'   footer, passing `...` (e.g. `max_rows` for the pjrt backend) on to the backend's printer.
#' @param jit (`function(f, static, cache_size, <options>, device = NULL)`)\cr
#'   Creates the backend's implementation of a JIT-compiled function and returns
#'   it as a `function`. The formals in place of `<options>` are the
#'   backend-specific options [`jit()`] accepts through `...`.
#' @param await_data (`function(x)`)\cr Blocks until the array's underlying data
#'   is ready. Called by [`await()`] for `AnvlArray`s; a no-op for backends
#'   without async execution.
#' @param default_dtypes (`NULL` | `list(float, int)`)\cr
#'   The default data types for this backend.
#'   Can be overwritten, see [`default_dtypes()`].
#' @return (`AnvlBackend`)
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
  platform_devices,
  copy_to_device,
  print_data,
  jit,
  await_data,
  default_dtypes
) {
  if (!is.null(default_dtypes)) {
    default_dtypes <- list(
      float = as_dtype(default_dtypes$float),
      int = as_dtype(default_dtypes$int)
    )
  }
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
      platform_devices = platform_devices,
      copy_to_device = copy_to_device,
      print_data = print_data,
      jit = jit,
      await_data = await_data,
      default_dtypes = default_dtypes
    ),
    class = "AnvlBackend"
  )
}

# `backend` is deliberately left unevaluated and built on first use.
#
# Registration happens as top-level code, which R evaluates when the namespace
# is *installed* and then restores from the lazy-load database on load. A
# backend constructed there would be a set of closures serialized before
# anything can instrument the namespace, which is why covr reported every
# backend method as untested however often the tests called it -- the methods
# it instruments and the ones the registry holds were different objects
# (r-lib/covr#556). A promise is forced by the first array operation instead,
# long after load, and builds its methods from whatever the namespace holds
# then. R forces it once and caches the value, so this costs nothing per call.
register_backend <- function(name, backend) {
  if (name %in% c("float", "int")) {
    cli_abort("A backend must not be named after a data type category ({.val float} or {.val int}).")
  }
  delayedAssign(name, backend, eval.env = environment(), assign.env = globals$backends)
}

# Compare two device objects for equality, returning FALSE when they are of
# different classes. Avoids R's "incompatible methods" warning when `==` is
# dispatched across device classes (e.g. PJRTDevice vs QuickrDevice).
eq_device <- function(x, y) {
  identical(class(x), class(y)) && isTRUE(x == y)
}

# Error when a traced graph closes over arrays from a backend other than
# `expected` (after accounting for `"plain"` constants, which are
# backend-agnostic), producing a clearer error than the downstream
# device-unification or codegen failures. The array inputs need no check here:
# the dispatcher already rejects one of another backend.
check_single_backend <- function(graph, expected) {
  const_backends <- vapply(
    graph$constants,
    function(const) if (is_concrete_array(const$aval)) backend(const$aval$data) else NA_character_,
    character(1L)
  )
  found <- unique(const_backends)
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

globals$backends <- new.env(parent = emptyenv())

# The plain backend is merely for capturing constants during jitting in a backend-agnostic way.
# Otherwise it is unused
register_backend(
  "plain",
  AnvlBackend(
    new_data = function(data, dtype, shape, device, row_major = FALSE) {
      if (is.raw(data)) {
        cli_abort("Raw {.arg data} payloads are not supported inside {.fn jit}.")
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
        substr(dtype_chr, 1L, 1L),
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
    # `new_data()` stores the data flat (coercing the storage mode drops `dim`)
    as_array = function(x, check) {
      if (length(x$shape) < 1L) {
        return(x$data)
      }
      array(x$data, dim = x$shape)
    },
    as_raw = function(x, row_major) cli_abort("as_raw not supported for plain backend"),
    platform = function(x) "cpu",
    device = function(x) PlainDeviceCpu(),
    new_device = function(type) {
      cli_abort("{.val plain} backend does not support creating devices.")
    },
    platform_devices = function(device) {
      cli_abort("{.val plain} backend does not support listing devices.")
    },
    copy_to_device = function(x, device) {
      cli_abort("{.val plain} backend does not support copying to a device.")
    },
    print_data = function(x, footer, ...) {
      print(x$data, ...)
      cat(footer, "\n")
    },
    jit = function(f, static, cache_size, ...) {
      cli_abort("JIT compilation is not supported for the {.val plain} backend.")
    },
    await_data = function(x) invisible(NULL),
    default_dtypes = NULL
  )
)

#' Get Active Backend
#'
#' Retrieves the active backend (option `anvl.backend`), falling back to the default `"pjrt"`
#' backend.
#'
#' @return (`character(1)`)\cr
#'   The backend name (e.g. `"pjrt"`, `"quickr"`).
#' @seealso [local_backend()], [with_backend()], [default_dtypes()]
#' @examples
#' active_backend()
#' @export
active_backend <- function() {
  getOption("anvl.backend", "pjrt")
}

# The backends a user can select: `"plain"` only holds constants during tracing.
assert_backend <- function(backend) {
  assert_choice(backend, setdiff(names(globals$backends), "plain"))
}

#' Temporarily Set the Backend
#'
#' Set the `anvl.backend` option for a scope: `local_backend()` until the
#' calling frame exits, `with_backend()` for the duration of `code`. Every
#' array built and every operation run in that scope uses the backend, and R
#' values materialize at its default data types (see [`default_dtypes()`]).
#'
#' @param backend (`character(1)`)\cr
#'   Backend to use (`"pjrt"` or `"quickr"`).
#' @param envir (`environment`)\cr
#'   The environment to scope the change to.
#' @param code (any)\cr
#'   An expression to evaluate with the given backend.
#' @return `local_backend()` returns the previous value of the option, as
#'   `list(anvl.backend = )`, invisibly. `with_backend()` returns the result
#'   of evaluating `code`.
#' @seealso [active_backend()]
#' @examplesIf requireNamespace("quickr", quietly = TRUE)
#' f <- function() {
#'   local_backend("quickr")
#'   active_backend()
#' }
#' f()
#' active_backend()
#' with_backend("quickr", active_backend())
#' @export
local_backend <- function(backend, envir = parent.frame()) {
  backend <- assert_backend(backend)
  withr::local_options(anvl.backend = backend, .local_envir = envir)
}

#' @rdname local_backend
#' @export
with_backend <- function(backend, code) {
  backend <- assert_backend(backend)
  withr::with_options(list(anvl.backend = backend), code)
}

#' Install What a Backend Needs to Run
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
#' For `"pjrt"`, the CPU plugin is always installed, and the CUDA plugin too
#' when an NVIDIA GPU is detected on Linux (or `cuda = TRUE` is passed). The
#' CUDA plugin additionally needs the CUDA libraries, which come in the
#' `pjrt.cuda` R package from the r-xla r-universe; `install_anvl()` installs
#' it along with the CUDA plugin. See [pjrt::install_pjrt()] for details.
#'
#' @param backend (`character(1)`)\cr
#'   Backend to install for. Defaults to [active_backend()].
#' @param ... Passed to the underlying installer: [pjrt::install_pjrt()] for
#'   `"pjrt"`, [utils::install.packages()] for `"quickr"`.
#' @return (`NULL`)\cr
#'   Invisibly. Called for its side effect.
#' @export
install_anvl <- function(backend = active_backend(), ...) {
  backend <- assert_choice(backend, c("pjrt", "quickr"))
  switch(
    backend,
    pjrt = pjrt::install_pjrt(...),
    quickr = install.packages("quickr", ...)
  )
  invisible(NULL)
}
