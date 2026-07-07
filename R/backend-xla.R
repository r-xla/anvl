#' @include backend.R
NULL

jit_call_xla <- function(
  exec,
  out_node,
  consts_flat,
  args_flat,
  is_static_flat,
  ambiguous_out = NULL,
  device,
  phantom_specs = list()
) {
  args_unwrapped <- lapply(args_flat[!is_static_flat], \(a) {
    if (is_valid_r_lit(a)) {
      pjrt_scalar(a, device = device)
    } else if (is_valid_r_array(a)) {
      pjrt_buffer(a, device = device)
    } else {
      # no-op if already on correct device
      pjrt::copy_buffer(a$data, device)
    }
  })

  # Phantom donated buffers for outputs that XLA was told to alias onto
  # extra-appended inputs. pjrt::pjrt_execute migrates the RAWSXP
  # keepalive from each phantom XPtr to its aliased output XPtr, so the
  # output's host bytes end up owned by R's GC.
  # We onlu use this on CPU, on CUDA phantom_specs is empty
  phantom_bufs <- lapply(phantom_specs, function(spec) {
    pjrt::pjrt_empty(dtype = spec$dtype, shape = spec$shape, device = device)
  })

  out_vals <- rlang::exec(
    pjrt::pjrt_execute,
    exec,
    !!!consts_flat,
    !!!args_unwrapped,
    !!!phantom_bufs,
    simplify = FALSE
  )
  jit_wrap_outputs(out_vals, out_node, ambiguous_out, "xla")
}

# device: NULL | PJRTDdevice | AnvlDeviceArg;
jit_xla_impl <- function(f, static, cache, donate, device) {
  if (!is.null(device) && !is_device_arg(device)) {
    device <- nv_device(device, "xla")
  }
  function() {
    if (currently_tracing()) {
      args <- as.list(match.call())[-1L]
      args <- lapply(args, eval, envir = parent.frame())
      return(do.call(f, args))
    }
    # - With specified device, inputs will be moved to it
    # - Otherwise checks that there are no conflicting devices
    # - it detects which device will be used (either select or inferred)
    prep <- jit_prepare_call(match.call(), parent.frame(), static, device = device, backend = "xla")
    avals_in <- to_avals(prep$args_flat, prep$is_static_flat)

    # prep$device is either
    # - device compatible with all inputs
    # - specified device
    # - NULL (device unknown from inputs)
    cache_key <- list(prep$in_tree, avals_in, prep$device)
    cache_hit <- cache$get(cache_key)

    if (!is.null(cache_hit)) {
      return(jit_call_xla(
        cache_hit[[1]], # executable
        cache_hit[[2]], # out tree
        cache_hit[[3]], # constants
        prep$args_flat,
        prep$is_static_flat,
        cache_hit[[4]], # ambiguity
        cache_hit[[5]], # device
        cache_hit[[6]] # phantom_specs
      ))
    }

    args_flat_nv <- prep$args_flat[!prep$is_static_flat & vapply(prep$args_flat, is_anvl_array, logical(1))]

    arg_devices <- lapply(args_flat_nv, tengen::device)

    # might still be NULL -> use default device
    compile_device <- if (is_device_arg(device)) {
      prep$args[[device$argname]]
    } else {
      device
    }
    compiled <- compile_xla(
      f,
      args_flat = avals_in,
      in_tree = prep$in_tree,
      donate = donate,
      device = compile_device,
      arg_devices = arg_devices
    )

    # Important: Here we pass compiled$device, which (if prep$device was NULL) is the
    # inferred device from the tracing (e.g. in jit(\() nv_scalar(1, device = "cpu:0")))
    out <- jit_call_xla(
      compiled$exec,
      compiled$out_tree,
      compiled$const_arrays,
      prep$args_flat,
      prep$is_static_flat,
      compiled$ambiguous_out,
      compiled$device,
      compiled$phantom_specs
    )

    cache$set(
      cache_key,
      list(
        compiled$exec,
        compiled$out_tree,
        compiled$const_arrays,
        compiled$ambiguous_out,
        compiled$device,
        compiled$phantom_specs
      )
    )

    return(out)
  }
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
#' @return A `list` with elements:
#'   - `exec`: The compiled PJRT executable.
#'   - `out_tree`: The output tree structure.
#'   - `const_arrays`: Constants needed at execution time.
#'   - `ambiguous_out`: Logical vector indicating which outputs are ambiguous (`NULL` if none are).
#' @keywords internal
compile_xla <- function(
  f,
  args_flat,
  in_tree,
  donate = character(),
  device = NULL,
  arg_devices = list()
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

  # jit() / xla() always run the full set of graph optimization passes.
  graph <- optimize_graph(graph, optimize = TRUE)

  # if device is NULL, all devices from args_flat and the traced devices must be the same.
  # If device is specified, then we use the requested device.

  unique_devices <- unique(c(desc$devices, arg_devices))

  if (is.null(device)) {
    # 0 input function and no allocated constants.
    # There might still be "plain" constants to be converted, so we
    # set device to default device
    if (length(unique_devices) == 0L) {
      device <- default_device("xla")
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

  ambiguous_out <- vapply(graph$outputs, \(x) x$aval$ambiguous, logical(1))
  if (!any(ambiguous_out)) {
    ambiguous_out <- NULL
  }

  src <- stablehlo::repr(func)
  program <- pjrt_program(src = src, format = "mlir")
  exec <- pjrt_compile(program, device = device)

  list(
    exec = exec,
    out_tree = out_tree,
    const_arrays = const_arrays,
    ambiguous_out = ambiguous_out,
    device = device(exec),
    phantom_specs = phantom_specs
  )
}

#' @title Ahead-of-time compile a function to XLA
#' @description
#' Compiles a function to an XLA executable via tracing.
#'
#' Returns a callable R function that executes the compiled binary.
#' Unlike [`jit()`], compilation happens eagerly at
#' definition time rather than on first call, so the input shapes and dtypes must be
#' specified upfront via abstract arrays (see [`nv_aval()`]).
#' @details
#' Traces `f` with the given abstract `args` (via [`trace_fn()`]), lowers the resulting graph
#' via [`stablehlo()`] and then compiles it to an XLA executable via [`pjrt::pjrt_compile()`].
#'
#' @param f (`function`)\cr
#'   Function to compile. Must accept and return [`AnvlArray`]s.
#' @param args (`list`)\cr
#'   List of abstract array specifications (e.g. from [`nv_aval()`]) describing the
#'   expected shapes and dtypes of `f`'s arguments.
#' @param donate (`character()`)\cr
#'   Names of the arguments whose buffers should be donated.
#' @param device (`character(1)` | `PJRTDevice`)\cr
#'   Target device such as `"cpu"` (default) or `"cuda"`.
#' @return (`function`)\cr
#'   A function that accepts [`AnvlArray`] arguments (matching the flat inputs)
#'   and returns the result as [`AnvlArray`]s.
#' @seealso [`jit()`] for lazy compilation, [`compile_xla()`] for the lower-level API.
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f_compiled <- xla(function(x, y) x + y,
#'   args = list(x = nv_aval("f32", c(2, 2)), y = nv_aval("f32", c(2, 2)))
#' )
#' a <- nv_matrix(1:4, nrow = 2, dtype = "f32")
#' b <- nv_matrix(5:8, nrow = 2, dtype = "f32")
#' f_compiled(a, b)
xla <- function(f, args, donate = character(), device = NULL) {
  # FIXME: Also use device inference from trace_fn
  device <- if (is.null(device)) {
    default_device("xla")
  } else {
    nv_device(device, "xla")
  }
  in_tree <- build_tree(args)
  args_flat <- flatten(args)
  compiled <- compile_xla(f, args_flat = args_flat, in_tree = in_tree, donate = donate, device = device)
  exec <- compiled$exec
  out_tree <- compiled$out_tree
  const_arrays <- compiled$const_arrays
  ambiguous_out <- compiled$ambiguous_out
  phantom_specs <- compiled$phantom_specs

  f_xla <- function() {
    prep <- jit_prepare_call(match.call(), parent.frame(), static = character(), device = device, backend = "xla")
    args_unwrapped <- lapply(prep$args_flat, \(a) {
      if (is_valid_r_lit(a)) {
        pjrt_scalar(a, device = device)
      } else if (is_valid_r_array(a)) {
        pjrt_buffer(a, device = device)
      } else {
        pjrt::copy_buffer(a$data, device)
      }
    })
    phantom_bufs <- lapply(phantom_specs, function(spec) {
      pjrt::pjrt_empty(dtype = spec$dtype, shape = spec$shape, device = device)
    })
    out_vals <- rlang::exec(
      pjrt::pjrt_execute,
      exec,
      !!!const_arrays,
      !!!args_unwrapped,
      !!!phantom_bufs,
      simplify = FALSE
    )
    jit_wrap_outputs(out_vals, out_tree, ambiguous_out, "xla")
  }
  formals(f_xla) <- formals2(f)
  f_xla
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
    new_data = function(data, dtype, shape, device, ambiguous) {
      buf <- pjrt_buffer(data, dtype = dtype, device = device, shape = shape)
      structure(
        list(data = buf, ambiguous = ambiguous, backend = "xla"),
        class = "AnvlArray"
      )
    },
    new_empty = function(dtype, shape, device, ambiguous) {
      buf <- pjrt::pjrt_empty(dtype = dtype, shape = shape, device = device)
      structure(
        list(data = buf, ambiguous = ambiguous, backend = "xla"),
        class = "AnvlArray"
      )
    },
    dtype = function(x) tengen::dtype(x$data),
    shape = function(x) tengen::shape(x$data),
    ambiguous = function(x) x$ambiguous,
    as_array = function(x, check) tengen::as_array(x$data, check = check),
    as_raw = function(x, row_major) tengen::as_raw(x$data, row_major = row_major),
    platform = function(x) pjrt::platform(x$data),
    device = function(x) device(x$data),
    new_device = function(x) pjrt::pjrt_device(x),
    print_data = function(x, footer) print(x$data, header = FALSE, footer = footer),
    jit = function(f, static, cache, donate = character(), device = NULL) {
      assert_subset(donate, formalArgs2(f))
      # static is checked in jit() itself
      common <- intersect(donate, static)
      if (length(common)) {
        cli_abort("{.val {common}} cannot be both in {.arg donate} and {.arg static}.")
      }
      jit_xla_impl(f, static, cache, donate, device)
    },
    await_data = function(x) pjrt::await(x$data)
  )
  class(backend) <- c("AnvlBackendXla", class(backend))
  backend
}

register_backend("xla", AnvlBackendXla())
