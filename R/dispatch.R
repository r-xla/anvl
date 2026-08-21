#' @title Dispatcher
#' @noRd
#' @description
#' A native fast path for repeatedly executing a compiled program on inputs of
#' the same signature: the engine behind [`jit()`]. It owns an executable cache
#' keyed on the inputs' structure and abstract values, calling back into R to
#' compile only on a cache miss.
#'
#' Internal. `jit()` is the only caller; the backends in `backend-pjrt.R` and
#' `backend-quickr.R` build one dispatcher each and supply the `compile`
#' callback.
#'
#' `dispatcher()` creates a dispatcher for one program family;
#' `dispatch()` runs a call through it and returns the call's result.
#'
#' @details
#' Each `dispatch()` call flattens the inputs and builds a cache key: a
#' dynamic leaf contributes its dtype, shape and `ambiguous` flag, a static leaf
#' its value (compared with [identical()]). On a hit the cached executable runs
#' immediately; on a miss `compile` is called to produce a new cache entry.
#'
#' Inputs are validated before the cache is probed, and a rejection names the
#' offending argument by its path in the argument tree. An input must be an
#' `"AnvlArray"` of the dispatcher's `backend`, a length-1 atomic scalar, or an
#' [is.array()] value. `compile` is therefore never asked to validate: it is
#' called only on a cache miss, and only for inputs already known to be
#' executable.
#'
#' @section Backends:
#' The `backend` selects the execution engine, and everything backend-specific
#' sits behind it:
#'
#' * `backend = "pjrt"` executes a compiled PJRT executable natively: array
#'   inputs contribute their `$data` buffer, bare R literals and arrays are
#'   uploaded with the same dtype defaults as
#'   [`pjrt_scalar()`]/[`pjrt_buffer()`], and the outputs are wrapped back into
#'   `"AnvlArray"`s -- lists of `$data`, `$dtype`, `$shape`, `$device`,
#'   `$ambiguous` and `$backend` -- and re-nested via `out_tree`, all without
#'   leaving C++.
#' * any other `backend` calls the compiled R closure `compile` returned, which
#'   returns the call's finished value. Execution, output wrapping and input
#'   placement therefore stay under the backend's control. This is the path for
#'   any non-PJRT backend (e.g. anvl's `"quickr"`).
#'
#' Of an array input, only `$data` is ever assumed: for `"pjrt"` its dtype, shape
#' and device are read off the `PJRTBuffer` directly, and for any other backend
#' they come from `extractor`. A backend is free to store them as fields or to
#' compute them on demand.
#'
#' @param capacity (`integer(1)`)\cr
#'   Maximum number of compiled executables to cache.
#' @param compile (`function`)\cr
#'   Cache-miss callback, called as `compile(info)`. `info` carries what the
#'   dispatch already derived from the call, so that the callback need not
#'   classify the inputs a second time:
#'   * `args`: the dispatched argument list,
#'   * `in_tree`: its `RTree` (see [`build_tree`]),
#'   * `leaves`: the flat leaf list (see [`flatten`]),
#'   * `is_static`: a `logical()` mask over `leaves`,
#'   * `avals`: per leaf, `NULL` if static, else the `list(dtype, shape,
#'     ambiguous)` the cache key was built from. `dtype` is a canonical dtype
#'     string (`"f32"`, `"i64"`, ...), `shape` an `integer()`, empty for a
#'     scalar,
#'   * `default_device`: the device this call resolved because no array input
#'     named one -- the device the cache key was built on, so `compile` must
#'     compile for it rather than resolve a default of its own. `NULL` when an
#'     array named the device, or under `move_inputs`.
#'
#'   For `backend = "pjrt"` it must return a named list with:
#'   * `exec`: a [`pjrt_compile`]d executable,
#'   * `client`, `device`: the [`pjrt_client`] and the device the entry is
#'     compiled for,
#'   * `out_tree`: the `RTree` of the outputs (see [`build_tree`]),
#'   * `out_avals`: one aval per output leaf of `out_tree`, each a
#'     `list(dtype = <string>, shape = <integer>, ambiguous = <logical(1)>)`
#'     (`ambiguous` is optional and defaults to `FALSE`). The outputs are
#'     wrapped from these.
#'   * `const_arrays` (optional): buffers prepended to the inputs,
#'   * `phantom_specs` (optional): a list of `list(dtype = <string>, shape =
#'     <integer>)` donation-output buffers to allocate fresh per call.
#'
#'   For any other `backend` it must return a named list with:
#'   * `r_fun`: a function called with the list of the call's dynamic leaves, in
#'     order and with an array leaf contributing its `$data`, returning the
#'     call's finished value. Static leaves are *not* passed: they are constants
#'     of the closure `compile` just built, and a cache hit already proves the
#'     call's statics are `identical()` to the ones it was built from.
#' @param static (`character()`)\cr
#'   Names of top-level arguments that are static (not arrays). Static values
#'   are part of the cache key and are excluded from execution. Defaults to
#'   none. They are compared with `identical(num.eq = FALSE)`, i.e. numbers
#'   compare bitwise, which is what keeps a `bit64::integer64` `NA` -- stored as
#'   the bit pattern of `-0` -- from sharing a cache entry with `0`.
#' @param backend (`character(1)`)\cr
#'   The `$backend` tag every `"AnvlArray"` input must carry, and the tag
#'   stamped on wrapped outputs. It also selects the execution engine (see
#'   *Backends*); anvl's quickr backend passes `"quickr"`.
#' @param move_inputs (`logical(1)`)\cr
#'   If `TRUE`, each cache entry has a target device -- the `device` its
#'   `compile` call returned -- and the engine places every input on it at
#'   execute time; the cache key then carries no device, so inputs may arrive
#'   from any device. Default `FALSE`: the first array's device is the call's
#'   device, and a conflicting input is an error.
#'
#'   It is a policy flag, not a device: *which* device an entry targets is the
#'   `compile` callback's business, and need not be the same for every entry
#'   (anvl's `jit(device = )` fixes one up front, its `device_arg()` derives one
#'   per static argument value).
#'
#'   Placing an input is the engine's business, since only it knows what `$data`
#'   holds. With `backend = "pjrt"` an input living elsewhere is copied to the
#'   entry's device. With any other backend pjrt does nothing, so **`r_fun` must
#'   place its own inputs** -- it receives only their `$data`, not their
#'   `$device`, so the placing has to be idempotent.
#' @param default_device (`function` | `NULL`)\cr
#'   Called with no arguments to get the backend's *current* default device,
#'   whenever a call has no array input to read a device from. Its result is
#'   part of the cache key, so an entry compiled under one default device is
#'   never served after the default changes. Required unless
#'   `move_inputs = TRUE`, which fixes the device per entry.
#'
#'   Devices are compared by object identity first and by [identical()] as a
#'   fallback, so equal-but-distinct device objects count as one device.
#'   Interning them (one object per device, alive for the session, as
#'   [`as_pjrt_device()`] does) is therefore not required, but is recommended:
#'   an interned device resolves in one pointer comparison, and every distinct
#'   object a backend hands out stays alive for the dispatcher's lifetime.
#' @param extractor (`function` | `NULL`)\cr
#'   Reads a non-`"pjrt"` array's metadata via the backend's accessors, called as
#'   `extractor(leaf)` and returning `list(aval = list(dtype, shape, ambiguous),
#'   device, backend)` -- `dtype` a tengen `DataType`, `shape` an `integer()`.
#'   Required for any backend other than `"pjrt"`; ignored for `"pjrt"` (see
#'   *Backends*).
#' @return `dispatcher()` returns a `Dispatcher`.
dispatcher <- function(
  capacity,
  compile,
  static = character(),
  backend = "pjrt",
  move_inputs = FALSE,
  default_device = NULL,
  extractor = NULL
) {
  checkmate::assert_count(capacity, positive = TRUE)
  checkmate::assert_function(compile)
  checkmate::assert_character(static, any.missing = FALSE)
  checkmate::assert_string(backend, min.chars = 1L)
  checkmate::assert_flag(move_inputs)
  checkmate::assert_function(default_device, null.ok = TRUE)
  checkmate::assert_function(extractor, null.ok = TRUE)
  if (!move_inputs && is.null(default_device)) {
    cli::cli_abort(
      "{.arg default_device} is required unless {.code move_inputs = TRUE}."
    )
  }
  # The backend fixes the execution engine: "pjrt" runs a compiled PJRT
  # executable natively; any other backend runs through the compiled R closure.
  engine <- if (backend == "pjrt") "pjrt" else "closure"
  if (engine == "closure" && is.null(extractor)) {
    cli::cli_abort(
      "{.arg extractor} is required for a non-{.val pjrt} backend: pass a
       function that reads a leaf's metadata via the backend's accessors."
    )
  }
  impl_dispatcher_create(
    as.integer(capacity),
    compile,
    as.character(static),
    engine,
    backend,
    move_inputs,
    default_device,
    extractor
  )
}

#' @title Native eager-dispatch fast path
#' @noRd
#' @description
#' Dispatch a call through a `dispatcher()`'s executable cache, compiling on a
#' miss. An input the dispatcher cannot execute is an error, named after the
#' offending argument; there is no fallback for the caller to take.
#' @param dispatcher (`Dispatcher`)\cr A dispatcher from `dispatcher()`.
#' @param args (`list`)\cr The (already evaluated) argument list of the call.
#' @return `dispatch()` returns the call's result. With `backend = "pjrt"` that
#'   is the output buffers wrapped into `"AnvlArray"`s and re-nested by the
#'   `compile` callback's `out_tree` (see `dispatcher()`); with any other
#'   backend it is whatever the compiled closure returned.
# Bound directly to the generated native entry (RcppExports.R precedes this
# file in the Collate order): dispatch() is on the per-primitive hot path, and
# a plain forwarding wrapper would add an R call frame per dispatch.
dispatch <- impl_dispatch_run

#' @title Number of compiled executables a dispatcher caches
#' @noRd
#' @return `dispatcher_size()` returns the number of compiled executables
#'   the dispatcher currently caches.
dispatcher_size <- impl_dispatcher_size

#' @export
print.Dispatcher <- function(x, ...) {
  cat("<Dispatcher>\n")
  invisible(x)
}
