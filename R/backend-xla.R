#' @include backend.R
NULL

# Build the native-dispatch compile callback for a jitted function. Invoked by
# pjrt's dispatcher only on a cache miss, and only for inputs the dispatcher has
# already validated -- so this traces, lowers, and compiles, and does no
# checking or classification of its own. `info` carries the tree, the flat
# leaves, the static mask, and the avals the cache key was built from.
# `device` is the jit's device policy: NULL (infer), a concrete device, or a
# device_arg() whose value is read from the static args.
jit_xla_compile_cb <- function(f, donate, device = NULL) {
  function(info) {
    compile_device <- if (is_device_arg(device)) {
      info$args[[device$argname]]
    } else {
      device
    }
    compiled <- compile_xla(
      f,
      args_flat = avals_from_dispatch(info),
      in_tree = info$in_tree,
      donate = donate,
      device = compile_device,
      arg_devices = dispatch_arg_devices(info),
      fallback_device = info$default_device
    )
    phantom_specs <- lapply(compiled$phantom_specs, function(spec) {
      list(dtype = as.character(spec$dtype), shape = as.integer(spec$shape))
    })
    list(
      exec = compiled$exec,
      const_arrays = compiled$const_arrays,
      phantom_specs = phantom_specs,
      client = pjrt::pjrt_client(pjrt::platform(compiled$device)),
      device = compiled$device,
      out_tree = compiled$out_tree,
      out_avals = compiled$out_avals
    )
  }
}

# device: NULL | PJRTDdevice | AnvlDeviceArg;
jit_xla_impl <- function(f, static, cache_size, donate, device) {
  if (!is.null(device) && !is_device_arg(device)) {
    device <- nv_device(device, "xla")
  }

  # pjrt's native dispatcher is the single cache + dispatch path. With a
  # target device (jit(device = ) or device_arg()) it copies buffer inputs to
  # the entry's device per call (move_inputs); otherwise the first input's
  # device is the call's device.
  dispatcher <- pjrt::dispatcher(
    cache_size,
    jit_xla_compile_cb(f, donate, device),
    static = static,
    move_inputs = !is.null(device),
    # Consulted only when a call has no array input to name a device. It reads
    # the default afresh, so a call of literals alone recompiles when the
    # default device changes rather than reusing the old entry.
    default_device = if (is.null(device)) function() default_device("xla")
  )
  # Hoisted out of the per-call path: `::` resolves via getExportedValue on
  # every evaluation, which costs ~1us per lookup.
  dispatch <- pjrt::dispatch

  # One call on already-evaluated args. This is the fast entry: jit_auto's
  # wrapper (which has already captured and evaluated the arguments) calls it
  # directly via the "jit_run_args" attribute, skipping the inner closure's
  # match.call() + eval() re-capture. The dispatcher validates the inputs
  # itself and errors on anything it cannot execute, so there is no fallback --
  # and it wraps the output buffers into AnvlArrays natively, so its result is
  # the call's result.
  run <- function(args) {
    dispatch(dispatcher, args)
  }

  fn <- function() {
    if (currently_tracing()) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    # Evaluate the args once; `run()` reuses them on the native path and the
    # fallback alike, so argument expressions with side effects are not
    # evaluated twice.
    args <- as.list(match.call())[-1L]
    args <- lapply(args, eval, envir = parent.frame())
    run(args)
  }
  attr(fn, "jit_run_args") <- run
  fn
}

#' @title Trace, lower, and compile a function to an XLA executable
#' @description
#' Takes a function, traces it into a computational graph, lowers it to StableHLO,
#' and compiles it to a PJRT executable. Returns the compiled executable along with
#' metadata needed for execution.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param args_flat (`list`)\cr
#'   Flat list of abstract input values.
#' @param in_tree (`Node`)\cr
#'   Tree structure of the inputs.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param device (`NULL` | `character(1)`)\cr
#'   Target device (e.g. `"cpu"`, `"cuda"`). If `NULL`, inferred from `arg_devices`
#'   and traced arrays.
#' @param arg_devices (`list`)\cr
#'   Devices of the concrete (non-static) input arguments, extracted before
#'   converting to abstract values. Used together with traced devices for
#'   device inference when `device` is `NULL`.
#' @param fallback_device (`NULL` | device)\cr
#'   The device to compile for when `device` is `NULL` and nothing in the graph
#'   names one. pjrt's dispatcher supplies the device it keyed the entry on, so
#'   the program and its cache key agree. `NULL` (a caller with no dispatcher in
#'   front of it) falls back to [`default_device()`].
#' @return A `list` with elements:
#'   - `exec`: The compiled PJRT executable.
#'   - `out_tree`: The output tree structure.
#'   - `const_arrays`: Constants needed at execution time.
#'   - `out_avals`: One `list(dtype, shape, ambiguous)` per output leaf; pjrt's
#'     dispatcher builds the output wrappers from these.
#' @keywords internal
compile_xla <- function(
  f,
  args_flat,
  in_tree,
  donate = character(),
  device = NULL,
  arg_devices = list(),
  fallback_device = NULL
) {
  desc <- local_descriptor()
  graph <- trace_fn(
    f,
    desc = desc,
    args_flat = args_flat,
    in_tree = in_tree,
    mode = "toplevel"
  )

  # Validate backends on the raw traced graph, before the optimization passes
  # below run: `inline_scalarish_constants()` inlines and drops scalar
  # constants, which would otherwise hide a closed-over constant from a foreign
  # backend and let `check_single_backend()` pass incorrectly.
  check_single_backend(graph, arg_devices, expected = "xla")

  # jit() always runs the full set of graph optimization passes.
  graph <- optimize_graph(graph, optimize = TRUE)

  # if device is NULL, all devices from args_flat and the traced devices must be the same.
  # If device is specified, then we use the requested device.

  unique_devices <- unique(c(desc$devices, arg_devices))

  if (is.null(device)) {
    # 0 input function and no allocated constants.
    # There might still be "plain" constants to be converted, so we
    # set device to default device
    if (length(unique_devices) == 0L) {
      # Nothing in the graph names a device. pjrt's dispatcher already resolved
      # one and keyed this entry on it, so compile for that exact device; only
      # a caller with no dispatcher in front of it falls back to the default.
      device <- fallback_device %||% default_device("xla")
    } else if (length(unique_devices) > 1L) {
      devices_str <- paste0(vapply(unique_devices, as.character, character(1)), collapse = ", ")
      cli_abort(c(
        "device is `NULL` (autodetect) but found more than one device",
        i = "Found devices: {devices_str}"
      ))
    } else {
      # only a single device
      device <- unique_devices[[1L]]
    }
  }
  # Otherwise, everything will be converted to requested device and it does not matter
  # If we found different devices during tracing.

  compile_graph_xla(graph, donate = donate, device = device)
}

compile_graph_xla <- function(graph, donate = character(), device) {
  platform_name <- if (is.character(device)) device else platform(device)
  out <- stablehlo(
    graph,
    donate = donate,
    donate_unaliased_outputs = TRUE,
    platform = platform_name
  )
  func <- out[[1L]]
  constants <- out[[2L]]
  phantom_specs <- out[[3L]]

  const_arrays <- lapply(constants, \(const) {
    if (!is_concrete_tensor(const$aval)) {
      cli_abort("Internal error: Not all constants are concrete arrays")
    }
    arr <- const$aval$data
    if (backend(arr) == "plain") {
      pjrt_buffer(as_array(arr), dtype = dtype(arr), device = device, shape = shape(arr))
    } else if (backend(arr) != "xla") {
      cli_abort("Found non-XLA constant in program")
    } else {
      # no copy is done if buffer already on correct device
      pjrt::copy_buffer(arr$data, device = device)
    }
  })

  out_tree <- graph$out_tree

  # pjrt needs these to create templates for the outputs
  # We also provide dtype and shape because these are created BEFORE the first execution
  out_avals <- lapply(graph$outputs, function(x) {
    list(
      dtype = as.character(x$aval$dtype),
      shape = shape(x$aval),
      ambiguous = x$aval$ambiguous
    )
  })

  src <- stablehlo::repr(func)
  program <- pjrt_program(src = src, format = "mlir")
  exec <- pjrt_compile(program, device = device)

  list(
    exec = exec,
    out_tree = out_tree,
    const_arrays = const_arrays,
    out_avals = out_avals,
    device = device(exec),
    phantom_specs = phantom_specs
  )
}

#' XLA backend
#'
#' Constructs the XLA backend, which stores array data in PJRT buffers (via
#' [`pjrt::pjrt_buffer()`]) and compiles jitted functions to XLA executables
#' via [`stablehlo()`] and [`pjrt::pjrt_compile()`]. This is the default
#' backend.
#'
#' @section Data representation:
#' An [`AnvlArray`] with `backend = "xla"` wraps a [`pjrt::pjrt_buffer()`]
#' stored in the `$data` field. The buffer owns the memory holding the tensor
#' values and may live on any device supported by PJRT (CPU, CUDA, Metal,
#' ...). Calling [`as_array()`] transfers the buffer contents back to an R
#' array; calling [`nv_array()`] on an R object uploads it to the requested
#' device.
#'
#' Each `AnvlArray` therefore has an associated device, queryable via
#' [`device()`]. A device is a [`pjrt::as_pjrt_device()`] object (e.g. the
#' platform `"cpu"` or `"cuda"`, optionally with an index such as `"cuda:1"`).
#' When `device` is `NULL` in [`nv_array()`] or the [`jit()`] wrapper, the
#' device defaults to the `PJRT_PLATFORM` environment variable (falling back
#' to `"cpu"`), or is inferred from the existing inputs of a jitted call.
#' Operations require all inputs to live on the same device.
#'
#' @section XLA JIT arguments:
#' * `donate` (`character()`, default `character()`): names of arguments whose
#'   underlying buffers may be donated to (i.e., reused/consumed by) the
#'   compiled XLA executable. Donated buffers must not be used again by the
#'   caller after the call; this can reduce memory usage and copies for large
#'   inputs. Must not overlap with `static`.
#'
#' @return An [`AnvlBackend`] object with subclass `"AnvlBackendXla"`.
#' @seealso [`AnvlBackend()`], [`AnvlBackendQuickr()`], [`local_backend()`], [`jit()`].
#' @export
AnvlBackendXla <- function() {
  backend <- AnvlBackend(
    # The dtype/shape/device of a PJRT buffer are immutable, so we resolve them
    # once here and cache them on the AnvlArray (as the plain/quickr backends
    # already do for dtype/shape). This turns the per-call dtype()/shape()/
    # device() reads on the hot dispatch path into plain field accesses instead
    # of repeated S3-dispatch -> C++/pjrt calls.
    new_data = function(data, dtype, shape, device, ambiguous) {
      buf <- pjrt_buffer(data, dtype = dtype, device = device, shape = shape)
      structure(
        list(
          data = buf,
          dtype = tengen::dtype(buf),
          shape = tengen::shape(buf),
          device = device(buf),
          ambiguous = ambiguous,
          backend = "xla"
        ),
        class = "AnvlArray"
      )
    },
    new_empty = function(dtype, shape, device, ambiguous) {
      buf <- pjrt::pjrt_empty(dtype = dtype, shape = shape, device = device)
      structure(
        list(
          data = buf,
          dtype = tengen::dtype(buf),
          shape = tengen::shape(buf),
          device = device(buf),
          ambiguous = ambiguous,
          backend = "xla"
        ),
        class = "AnvlArray"
      )
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    ambiguous = function(x) x$ambiguous,
    as_array = function(x, check) tengen::as_array(x$data, check = check),
    as_raw = function(x, row_major) tengen::as_raw(x$data, row_major = row_major),
    platform = function(x) pjrt::platform(x$data),
    device = function(x) x$device,
    new_device = function(x) pjrt::pjrt_device(x),
    print_data = function(x, footer) print(x$data, header = FALSE, footer = footer),
    jit = function(f, static, cache_size, donate = character(), device = NULL) {
      assert_subset(donate, formalArgs2(f))
      # static is checked in jit() itself
      common <- intersect(donate, static)
      if (length(common)) {
        cli_abort("{.val {common}} cannot be both in {.arg donate} and {.arg static}.")
      }
      jit_xla_impl(f, static, cache_size, donate, device)
    },
    await_data = function(x) pjrt::await(x$data)
  )
  class(backend) <- c("AnvlBackendXla", class(backend))
  backend
}

register_backend("xla", AnvlBackendXla())
