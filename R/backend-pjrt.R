#' @include backend.R
NULL

# Build the native-dispatch compile callback for a jitted function. Invoked by
# pjrt's dispatcher only on a cache miss, and only for inputs the dispatcher has
# already validated -- so this traces, lowers, and compiles, and classifies
# nothing itself. The one check it does make is check_static_args(): a cache
# miss is where a static value is first used, and pjrt cannot know which values
# anvl considers sound to key on. `info` carries the tree, the flat leaves, the
# static mask, and the avals the cache key was built from.
# `device` is the jit's device policy: NULL (infer) or a concrete device.
jit_pjrt_compile_cb <- function(f, static, donate, xla_options, device = NULL) {
  function(info) {
    check_static_args(info$args, static)
    compiled <- compile_pjrt(
      f,
      args_flat = avals_from_dispatch(info),
      in_tree = info$in_tree,
      donate = donate,
      xla_options = xla_options,
      device = device,
      arg_device = dispatch_arg_device(info),
      fallback_device = info$default_device,
      default_dtypes = default_dtypes_from_key(info$context)
    )
    phantom_specs <- lapply(compiled$phantom_specs, function(spec) {
      list(dtype = as.character(spec$dtype), shape = as.integer(spec$shape))
    })
    list(
      exec = compiled$exec,
      const_arrays = compiled$const_arrays,
      input_dtypes = compiled$input_dtypes,
      phantom_specs = phantom_specs,
      client = pjrt::pjrt_client(pjrt::platform(compiled$device)),
      device = compiled$device,
      out_tree = compiled$out_tree,
      out_avals = compiled$out_avals,
      implicit_inputs = compiled$implicit_inputs,
      on_success = compiled$on_success,
      uses_default_device = compiled$uses_default_device
    )
  }
}

# device: NULL | PJRTDevice
jit_pjrt_impl <- function(f, static, cache_size, donate, xla_options, device) {
  if (!is.null(device)) {
    device <- backend_device(device, "pjrt")
  }

  # pjrt's native dispatcher is the single cache + dispatch path. With a
  # target device (jit(device = )) it copies buffer inputs to the entry's
  # device per call (move_inputs); otherwise the first input's device is the
  # call's device.
  dispatcher <- pjrt::dispatcher(
    cache_size,
    jit_pjrt_compile_cb(f, static, donate, xla_options, device),
    static = static,
    # The tag every AnvlArray input must carry and that the dispatcher stamps
    # on the arrays it returns; it also selects pjrt's native execution engine.
    backend = "pjrt",
    move_inputs = !is.null(device),
    # Consulted only when a call has no array input to name a device. It reads
    # the default afresh, so a call of literals alone recompiles when the
    # default device changes rather than reusing the old entry.
    default_device = if (is.null(device)) function() default_device("pjrt"),
    # Buffers hold the reversed shape (see R/lowering-axes.R): the dispatcher
    # reports and declares R shapes and uploads R data without reordering it.
    reverse_shapes = TRUE,
    # The default dtypes the program is compiled under are part of the key, so
    # a program compiled under one pair is never served under another.
    context = default_dtypes_context("pjrt")
  )
  # Hoisted out of the per-call path: `::` resolves via getExportedValue on
  # every evaluation, which costs ~1us per lookup.
  dispatch <- pjrt::dispatch

  # One call on already-evaluated args. This is the fast entry: jit()'s
  # wrapper (which has already captured and evaluated the arguments) calls it
  # directly via the "jit_run_args" attribute, skipping the inner closure's
  # match.call() + eval() re-capture. The dispatcher validates the inputs
  # itself and errors on anything it cannot execute, and it wraps the output
  # buffers into AnvlArrays natively, so its result is the call's result.
  run <- function(args) {
    dispatch(dispatcher, args)
  }

  fn <- function() {
    args <- lapply(as.list(match.call())[-1L], eval, envir = parent.frame())
    # Inside a trace, the call is inlined into the enclosing graph.
    if (currently_tracing()) {
      return(do.call(f, args))
    }
    run(args)
  }
  attr(fn, "jit_run_args") <- run
  fn
}

#' @title Trace, Lower, and Compile a Function to an XLA Executable
#' @description
#' Takes a function, traces it into a computational graph, lowers it to StableHLO,
#' and compiles it to a PJRT executable. Returns the compiled executable along with
#' metadata needed for execution.
#' @param f (`function`)\cr
#'   Function to compile.
#' @param args_flat (`list`)\cr
#'   Flat list of abstract input values.
#' @param in_tree ([`RTree`][pjrt::build_tree])\cr
#'   Tree structure of the inputs.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param xla_options (named `list()`)\cr
#'   XLA options for this compile, passed to [`pjrt::pjrt_compile()`].
#' @param device (`NULL` | `character(1)` | `PJRTDevice`)\cr
#'   Target device (e.g. `"cpu"`, `"cuda"`). If `NULL`, inferred from
#'   `arg_device` and traced arrays.
#' @param arg_device (`NULL` | device)\cr
#'   The device the array inputs live on, or `NULL` when there are none. Used
#'   together with traced devices for device inference when `device` is `NULL`.
#' @param default_dtypes (`NULL` | `list(float, int)`)\cr
#'   The data types the traced R values materialize at when nothing else decides
#'   one (see [`default_dtypes()`]).
#' @param fallback_device (`NULL` | device)\cr
#'   The device to compile for when `device` is `NULL` and nothing in the graph
#'   names one. pjrt's dispatcher supplies the device it keyed the entry on, so
#'   the program and its cache key agree. `NULL` (a caller with no dispatcher in
#'   front of it) falls back to [`default_device()`].
#' @return (`list`)\cr
#'   With elements:
#'   - `exec`: The compiled PJRT executable.
#'   - `out_tree`: The output tree structure.
#'   - `const_arrays`: Constants needed at execution time.
#'   - `out_avals`: One `list(dtype, shape)` per output leaf; pjrt's
#'     dispatcher builds the output wrappers from these.
#'   - `input_dtypes`: One entry per input: the dtype an input built from bare
#'     R data is uploaded at, and `NA` for an array input, which is supplied as
#'     it is. The R data has no dtype of its own, so the program is the only
#'     thing that knows what it is uploaded as -- pjrt's dispatcher therefore
#'     requires an entry for every bare R input and rejects a dtype declared
#'     for an array one. `NULL` for a call whose inputs are all arrays.
#'   - `device`: The `PJRTDevice` the executable was compiled for.
#'   - `phantom_specs`: One `list(dtype, shape)` per phantom donated input the
#'     executor must allocate for an output (see [`stablehlo()`]).
#'   - `implicit_inputs`: One `list(get, dtype, shape)` per value of a side
#'     effect the program takes from outside of it, which pjrt's dispatcher
#'     supplies as an input after the call's own: `get()` takes no arguments
#'     and returns the current value, a `PJRTBuffer` of `dtype` and `shape`,
#'     e.g. the key of the global random state.
#'   - `on_success`: `NULL`, or a function pjrt's dispatcher calls after every
#'     successful run with the call's arguments: it advances the host object
#'     of each side effect by its count, e.g. a random state passed as an
#'     argument.
#'   - `uses_default_device`: Whether the program was compiled for
#'     `fallback_device` (or the default device) because nothing else named
#'     one. `FALSE` means its device is fixed by `device`, the inputs or the
#'     traced arrays, which the default device does not affect.
#' @keywords internal
compile_pjrt <- function(
  f,
  args_flat,
  in_tree,
  donate = character(),
  xla_options = list(),
  device = NULL,
  arg_device = NULL,
  fallback_device = NULL,
  default_dtypes = NULL
) {
  desc <- local_descriptor(default_dtypes = default_dtypes, backend = "pjrt", side_effects_mode = "program")
  graph <- trace_fn(
    f,
    desc = desc,
    args_flat = args_flat,
    in_tree = in_tree
  )

  # Validate backends on the raw traced graph, before the optimization passes
  # below run: `inline_scalarish_constants()` inlines and drops scalar
  # constants, which would otherwise hide a closed-over constant from a foreign
  # backend and let `check_single_backend()` pass incorrectly.
  check_single_backend(graph, expected = "pjrt")

  # jit() always runs the full set of graph optimization passes.
  graph <- optimize_graph(graph, optimize = TRUE)

  # A requested device is used as is: everything is moved to it, whatever
  # devices the trace found. Otherwise the array inputs and the traced arrays
  # must all agree on one.
  uses_default_device <- FALSE
  if (is.null(device)) {
    unique_devices <- unique(c(desc$devices, if (!is.null(arg_device)) list(arg_device)))
    if (length(unique_devices) == 0L) {
      # Nothing in the graph names a device (though "plain" constants may still
      # need one). pjrt's dispatcher already resolved one and keyed this entry
      # on it, so compile for that exact device; only a caller with no
      # dispatcher in front of it falls back to the default.
      device <- fallback_device %||% default_device("pjrt")
      uses_default_device <- TRUE
    } else if (length(unique_devices) > 1L) {
      devices_str <- paste0(vapply(unique_devices, as.character, character(1L)), collapse = ", ")
      cli_abort(c(
        "device is `NULL` (autodetect) but found more than one device",
        i = "Found devices: {devices_str}"
      ))
    } else {
      device <- unique_devices[[1L]]
    }
  }

  out <- compile_graph_pjrt(graph, donate = donate, device = device, xla_options = xla_options)
  # Sound because the trace never reads the default device: an array created
  # without one while tracing is "plain" and placed on the compile device.
  out$uses_default_device <- uses_default_device
  out
}

# pjrt's dispatcher supplies the side effects the program takes as implicit
# inputs, and calls the entry's `on_success` after a run.
compile_graph_pjrt <- function(graph, donate = character(), device, xla_options = list()) {
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
    if (!is_concrete_array(const$aval)) {
      cli_abort("Internal error: Not all constants are concrete arrays")
    }
    arr <- const$aval$data
    if (backend(arr) == "plain") {
      pjrt_buffer(
        as_array(arr),
        dtype = dtype(arr),
        device = device,
        shape = rev(shape(arr)),
        row_major = TRUE,
        check = arr$check
      )
    } else if (backend(arr) != "pjrt") {
      cli_abort("Found non-PJRT constant in program")
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
      shape = shape(x$aval)
    )
  })

  # the functions `func` calls are defined next to it
  src <- paste(vapply(c(out[[4L]], list(func)), stablehlo::repr, character(1L)), collapse = "\n\n")
  program <- pjrt_program(src = src, format = "mlir")
  exec <- pjrt_compile(program, device = device, xla_options = xla_options)

  list(
    exec = exec,
    out_tree = out_tree,
    const_arrays = const_arrays,
    out_avals = out_avals,
    # The dtype each input is supplied at. Only an input built from bare R data
    # names one -- the dispatcher uploads the R data at that dtype, which is how
    # an `f64` program gets the exact R double; an array input takes `NA`, and a
    # call without any R input needs no declaration at all.
    input_dtypes = graph_input_dtypes(graph),
    device = device(exec),
    phantom_specs = phantom_specs,
    implicit_inputs = side_effects_implicit_inputs(graph),
    on_success = side_effects_on_success(graph)
  )
}

#' PJRT Backend
#'
#' Constructs the PJRT backend, which stores array data in PJRT buffers (via
#' [`pjrt::pjrt_buffer()`]) and compiles jitted functions to XLA executables
#' via [`stablehlo()`] and [`pjrt::pjrt_compile()`]. This is the default
#' backend.
#'
#' @section Data representation:
#' An [`AnvlArray`] with `backend = "pjrt"` wraps a [`pjrt::pjrt_buffer()`]
#' stored in the `$data` field. The buffer owns the memory holding the array
#' values and may live on any device supported by PJRT (CPU, CUDA, ...). It
#' holds the array's column-major bytes as an XLA array of the reversed shape,
#' so that uploads and read-backs copy the bytes without reordering them.
#' Calling [`as_array()`] transfers the buffer contents back to an R
#' array; calling [`nv_array()`] on an R object uploads it to the requested
#' device.
#'
#' Each `AnvlArray` therefore has an associated device, queryable via
#' [`device()`]. A device is a [`pjrt::as_pjrt_device()`] object (e.g. the
#' platform `"cpu"` or `"cuda"`, optionally with an index such as `"cuda:1"`).
#' When `device` is `NULL` in [`nv_array()`] or the [`jit()`] wrapper, the
#' device defaults to [`default_device()`], or is inferred from the existing
#' inputs of a jitted call.
#' Operations require all inputs to live on the same device.
#'
#' @section Supported data types:
#' `bool`; the signed integers `i8`, `i16`, `i32` and `i64`; the unsigned
#' integers `ui8`, `ui16`, `ui32` and `ui64`; and the floats `f32` and `f64`.
#' An R double materializes at `f32` on this backend and an R integer at `i32`
#' unless the defaults say otherwise (see [`default_dtypes()`]).
#'
#' @section Floating-point behavior:
#' Subnormal floating-point values may be preserved when stored in an array and
#' read back into R, yet treated as zero in calculations. On CPUs, XLA enables
#' a mode that replaces subnormal inputs and results with zero. The exact
#' behavior depends on the platform, backend, and operation.
#'
#' See `r roxy_article("gotchas")` for an explanation and examples.
#'
#' @return ([`AnvlBackend`])\cr
#'   With subclass `"AnvlBackendPjrt"`.
#' @seealso [`AnvlBackend()`], [`AnvlBackendQuickr()`], [`local_backend()`], [`jit()`].
#' @export
AnvlBackendPjrt <- function() {
  backend <- AnvlBackend(
    new_data = function(data, dtype, shape, device, row_major = FALSE, check = NULL) {
      if (inherits(data, "PJRTBuffer")) {
        return(pjrt_anvl_array_of_buffer(data, dtype, shape, device))
      }
      # Everything but a buffer is placed on the default device when the call
      # names none.
      device <- device %||% default_device("pjrt")
      # R data is column-major, which is the row-major order of the reversed
      # shape, so it is copied as it is. Only a raw payload can come in
      # row-major order (`byrow`), and that is the column-major order of the
      # reversed shape.
      row_major_rev <- if (is.raw(data)) !row_major else TRUE
      pjrt_anvl_array(
        pjrt_buffer(data, dtype = dtype, device = device, shape = rev(shape), row_major = row_major_rev, check = check),
        shape = as.integer(shape)
      )
    },
    new_empty = function(dtype, shape, device) {
      buf <- pjrt::pjrt_empty(dtype = dtype, shape = rev(shape), device = device %||% default_device("pjrt"))
      pjrt_anvl_array(buf, shape = as.integer(shape))
    },
    dtype = function(x) x$dtype,
    shape = function(x) x$shape,
    as_array = function(x, check) pjrt::as_array(x$data, check = check, row_major = TRUE),
    as_raw = function(x, row_major) xlamisc::as_raw(x$data, row_major = !row_major),
    platform = function(x) pjrt::platform(x$data),
    device = function(x) xlamisc::device(x$data),
    new_device = function(x) pjrt::pjrt_device(x),
    print_data = function(x, footer, ...) print(x$data, header = FALSE, footer = footer, reversed = TRUE, ...),
    jit = function(f, static, cache_size, donate = character(), xla_options = list(), device = NULL) {
      assert_subset(donate, formalArgs2(f))
      assert_list(xla_options)
      # static is checked in jit() itself
      common <- intersect(donate, static)
      if (length(common)) {
        cli_abort("{.val {common}} cannot be both in {.arg donate} and {.arg static}.")
      }
      jit_pjrt_impl(f, static, cache_size, donate, xla_options, device)
    },
    await_data = function(x) pjrt::await(x$data),
    default_dtypes = list(float = "f32", int = "i32")
  )
  class(backend) <- c("AnvlBackendPjrt", class(backend))
  backend
}

# Wrap a PJRT buffer that holds an array in anvl's storage -- its column-major
# bytes as an XLA array of the reversed shape -- into an AnvlArray. The
# dtype/shape of a buffer are immutable, so they are resolved once here and
# cached on the AnvlArray (as the plain/quickr backends already do). This turns
# the per-call dtype()/shape() reads on the hot dispatch path into plain field
# accesses instead of repeated S3-dispatch -> C++/pjrt calls. The device is not
# cached: a PJRTDevice is an external pointer, which would not survive
# serializing the array, while the buffer itself does. A caller that knows the
# array's `dtype` (a `DataType`) and R `shape` passes them, which saves reading
# them.
pjrt_anvl_array <- function(buf, dtype = xlamisc::dtype(buf), shape = rev(xlamisc::shape(buf))) {
  structure(
    list(
      data = buf,
      dtype = dtype,
      shape = shape,
      backend = "pjrt"
    ),
    class = "AnvlArray"
  )
}

register_backend("pjrt", AnvlBackendPjrt())

# The AnvlArray of a bare PJRTBuffer handed to `nv_array()`. Nothing on a buffer
# says which layout its bytes are in -- a buffer of shape (3, 2) is pjrt's 3x2
# array and anvl's 2x3 one -- so a bare buffer is read the way pjrt reads it: its
# shape is the array's shape and `as_array(buf)` its values. Code that holds a
# buffer already in anvl's layout wraps it with `pjrt_anvl_array()` instead.
# `pjrt_buffer()` checks that `dtype`, `shape` and `device`, when given, agree
# with the buffer's. anvl stores the reversed shape, so a buffer of rank 2 or
# more is read back column-major and uploaded again; a buffer of rank 0 or 1 is
# stored as it is.
pjrt_anvl_array_of_buffer <- function(buf, dtype, shape, device) {
  buf <- pjrt_buffer(buf, dtype = dtype, device = device, shape = shape)
  buf_shape <- xlamisc::shape(buf)
  buf_dtype <- xlamisc::dtype(buf)
  if (length(buf_shape) <= 1L) {
    return(pjrt_anvl_array(buf, dtype = buf_dtype, shape = buf_shape))
  }
  stored <- pjrt_buffer(
    xlamisc::as_raw(buf, row_major = FALSE),
    dtype = as.character(buf_dtype),
    shape = rev(buf_shape),
    device = xlamisc::device(buf),
    row_major = TRUE
  )
  pjrt_anvl_array(stored, dtype = buf_dtype, shape = buf_shape)
}
