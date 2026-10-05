#' @include device.R
NULL

#' @title Map a Function over Devices in Parallel
#' @description
#' Like [lapply()], calls `.f(.x[[i]], ...)` for every element of `.x`, but
#' runs each call on a device of its own, with the devices computing in
#' parallel.
#'
#' Its use is work that splits into independent pieces of the same shape, such
#' as the chains of an MCMC sampler: the calls do not communicate, and each
#' runs the program it would run on its own.
#'
#' @section Devices:
#' The elements of `.x` are assigned to `.devices` in turn: element `i` runs on
#' device `(i - 1) %% length(.devices) + 1`. Each round of up to
#' `length(.devices)` elements is one launch of a single program, compiled once
#' for all the round's devices; with more elements than devices, the rounds run
#' one after another.
#'
#' Every array in `.x[[i]]` and in `...` is copied to the call's device. Arrays
#' that `.f` captures from its environment are not: pass them through `...`
#' instead.
#'
#' To run on several CPU devices, the CPU client has to be created with them:
#' set the environment variable `PJRT_CPU_DEVICE_COUNT` (e.g.
#' `Sys.setenv(PJRT_CPU_DEVICE_COUNT = 4L)`) before the first array is created
#' or the first jitted function is called in the session. The count cannot be
#' changed afterwards without restarting R.
#'
#' On a backend that runs on a single device, such as quickr, the calls run one
#' after another.
#'
#' @param .x (`list` | `vector`)\cr
#'   The elements to map over. Every element must give `.f` arguments of the
#'   same structure, data types and shapes, and the same values for its static
#'   arguments, so that all calls run the same program.
#' @param .f (`function`)\cr
#'   Called as `.f(.x[[i]], ...)`. Typically a [jit()]-compiled function, whose
#'   `static` arguments are honoured and which keeps the compiled program across
#'   calls of `device_map()`. A plain function is compiled for the duration of
#'   one `device_map()` call, with no static arguments.
#' @param ... (`any`)\cr
#'   Further arguments to `.f`, the same for every call. Arrays among them are
#'   copied once to each device.
#' @param .devices (`NULL` | `character()` | `list()` of devices)\cr
#'   The devices to run on, of the active backend. The default (`NULL`) is every
#'   device of the platform of the [default device][default_device].
#' @return (`list`)\cr
#'   Of the same length and with the same names as `.x`: element `i` is the
#'   result of `.f(.x[[i]], ...)`, with its arrays on the device the call ran
#'   on. `device_map()` returns as soon as the calls are launched; reading an
#'   array (e.g. with [as_array()]) waits for it, as does [await()].
#' @seealso [jit()], [default_device()], [nv_device()]
#' @examplesIf pjrt::plugins_downloaded()
#' f <- jit(function(seed, n) nv_rnorm(n, nv_rng_state(seed))$values, static = "n")
#' # one sample per seed, each drawn on a device of its own
#' samples <- device_map(1:4, f, n = 3L)
#' samples[[1L]]
#' @export
device_map <- function(.x, .f, ..., .devices = NULL) {
  if (currently_tracing()) {
    cli_abort(c(
      "{.fn device_map} cannot be called inside a {.fn jit}-compiled function.",
      i = "A compiled program runs on one device; call {.fn device_map} on the jitted function instead."
    ))
  }
  if (is_anvl_array(.x)) {
    cli_abort(c(
      "{.arg .x} must be a list or a vector, not an array.",
      i = "Wrap it in a list to map over it as one element: {.code list(x)}."
    ))
  }
  assert_function(.f, .var.name = ".f")
  devices <- resolve_map_devices(.devices)
  .x <- as.list(.x)

  # The shared arguments are copied to each device once, not once per call.
  dots <- list(...)
  dots_on <- lapply(devices, function(device) to_device(dots, device))
  device_of <- function(i) (i - 1L) %% length(devices) + 1L
  args_of <- function(i) {
    k <- device_of(i)
    c(list(to_device(.x[[i]], devices[[k]])), dots_on[[k]])
  }

  out <- if (active_backend() == "pjrt") {
    device_map_pjrt(.x, .f, args_of, devices)
  } else {
    lapply(seq_along(.x), function(i) {
      with_default_device(devices[[device_of(i)]], do.call(.f, args_of(i)))
    })
  }
  names(out) <- names(.x)
  out
}

# `.devices` of `device_map()` as a list of devices of the active backend.
resolve_map_devices <- function(devices) {
  backend <- active_backend()
  if (is.null(devices)) {
    return(globals$backends[[backend]]$platform_devices(default_device(backend)))
  }
  if (is_device(devices) || is.character(devices)) {
    devices <- if (is_device(devices)) list(devices) else as.list(devices)
  }
  checkmate::assert_list(devices, min.len = 1L, .var.name = ".devices")
  lapply(devices, backend_device, backend = backend)
}

# `x` with every array in it -- itself, or a leaf of a (nested) list -- copied
# to `device`. Arrays already there are returned as they are.
to_device <- function(x, device) {
  if (is_anvl_array(x)) {
    if (eq_device(device(x), device)) {
      return(x)
    }
    return(globals$backends[[backend(x)]]$copy_to_device(x, device))
  }
  if (is.list(x) && !is.object(x)) {
    return(lapply(x, to_device, device = device))
  }
  x
}

# The pjrt implementation of `device_map()`: each round of up to
# `length(devices)` elements is one launch of a replicated executable, whose
# replicas XLA runs in parallel, each on a thread of its own device. Calling
# the single-device executables one after another would not overlap: the CPU
# client runs a program it estimates to be cheap -- which includes a whole
# loop whose body is -- on the calling thread, so each call would block.
device_map_pjrt <- function(.x, .f, args_of, devices) {
  cfg <- jit_config(.f)
  if (!is.null(cfg$device)) {
    cli_abort(c(
      "{.arg .f} must not be compiled for a device of its own.",
      i = "{.fn device_map} places the calls itself; drop {.arg device} from the {.fn jit} call."
    ))
  }
  f <- cfg$f %||% .f
  static <- cfg$static %||% character()
  cache <- device_map_cache(.f)

  calls <- lapply(seq_along(.x), function(i) map_call(f, args_of(i), static))
  if (!length(calls)) {
    return(list())
  }
  key <- calls[[1L]]$key
  for (i in seq_along(calls)) {
    if (!identical(calls[[i]]$key, key)) {
      cli_abort(c(
        "Every element of {.arg .x} must give {.arg .f} arguments of the same structure, data types and shapes.",
        x = "Element {i} differs from element 1.",
        i = "All calls run one program; {.fn lapply} runs calls that differ."
      ))
    }
  }

  rounds <- split(seq_along(calls), (seq_along(calls) - 1L) %/% length(devices))
  out <- vector("list", length(calls))
  for (round in rounds) {
    round_devices <- devices[seq_along(round)]
    compiled <- device_map_compile(cache, f, calls[[1L]], round_devices)
    inputs <- Map(
      function(call, device) map_call_inputs(call, compiled, device),
      calls[round],
      round_devices
    )
    outputs <- pjrt::pjrt_execute_replicated(compiled$exec, inputs, check = FALSE)
    out[round] <- lapply(outputs, function(bufs) {
      unflatten(compiled$out_tree, lapply(bufs, new_pjrt_array))
    })
  }
  out
}

# One call of `device_map()`: its leaves, which of them are static, their
# abstract values, and the key that tells whether two calls run one program.
map_call <- function(f, args, static) {
  args <- match_args_to_formals(f, args)
  in_tree <- build_tree(args)
  leaves <- flatten(args)
  is_static <- if (length(static)) pjrt::tree_leaf_mask(in_tree, static) else rep(FALSE, length(leaves))
  avals <- Map(
    function(leaf, static_leaf, j) {
      if (static_leaf) {
        return(leaf)
      }
      if (is_anvl_array(leaf)) {
        return(nv_aval(dtype(leaf), shape(leaf)))
      }
      if (is_map_rdata(leaf)) {
        return(nv_aval(typeof(leaf), as.integer(dim(leaf))))
      }
      cli_abort(c(
        "Input {j} of the call is neither an array nor an R number.",
        i = "Declare the argument it belongs to {.arg static} in {.fn jit}."
      ))
    },
    leaves,
    is_static,
    seq_along(leaves)
  )
  # The program depends on the structure, the abstract values and the static
  # values, so the key is the tree rebuilt from those.
  key_leaves <- Map(
    function(av, static_leaf) {
      if (static_leaf) {
        list(av)
      } else if (is_rdata(av)) {
        list("rdata", av$r_type, shape(av))
      } else {
        list("array", as.character(av$dtype), shape(av))
      }
    },
    avals,
    is_static
  )
  list(
    leaves = leaves,
    is_static = is_static,
    avals = avals,
    in_tree = in_tree,
    key = rlang::hash(unflatten(in_tree, key_leaves))
  )
}

# Bare R data the way pjrt's dispatcher accepts it as an input: an unclassed
# double, integer or logical array, or a scalar that is not `NA`.
is_map_rdata <- function(x) {
  is.atomic(x) &&
    !is.object(x) &&
    typeof(x) %in% c("double", "integer", "logical") &&
    (!is.null(dim(x)) || (length(x) == 1L && (!is.na(x) || is.nan(x))))
}

# The replicated executable for `call` on `devices`, compiled on first use and
# kept in `cache`.
device_map_compile <- function(cache, f, call, devices) {
  dtypes <- current_default_dtypes()
  key <- rlang::hash(list(
    call$key,
    vapply(devices, as.character, character(1L)),
    as.character(dtypes$float),
    as.character(dtypes$int)
  ))
  compiled <- cache[[key]]
  if (is.null(compiled)) {
    compiled <- compile_pjrt(
      f,
      args_flat = call$avals,
      in_tree = call$in_tree,
      device = devices,
      default_dtypes = dtypes
    )
    cache[[key]] <- compiled
  }
  compiled
}

# Where `device_map()` keeps the programs it compiled for `f`: on a jitted
# function, next to its own cache, so that they live as long as it does; for
# a plain function, for the one call.
device_map_cache <- function(f) {
  if (!inherits(f, "JitFunction")) {
    return(new.env(parent = emptyenv()))
  }
  env <- environment(f)
  if (is.null(env$.jit_device_map)) {
    env$.jit_device_map <- new.env(parent = emptyenv())
  }
  env$.jit_device_map
}

# The executable's inputs for one call on `device`, in the order pjrt's
# dispatcher assembles them: the constants, the call's inputs -- an array
# copied to `device`, an R value uploaded at the data type the program takes it
# at -- and the donation buffers to allocate.
map_call_inputs <- function(call, compiled, device) {
  consts <- lapply(compiled$const_arrays, pjrt::copy_buffer, device = device)
  leaves <- call$leaves[!call$is_static]
  avals <- call$avals[!call$is_static]
  inputs <- Map(
    function(leaf, av, dtype) {
      if (is_anvl_array(leaf)) {
        return(pjrt::copy_buffer(leaf$data, device = device))
      }
      pjrt_buffer(leaf, dtype = dtype, device = device, shape = shape(av))
    },
    leaves,
    avals,
    compiled$input_dtypes %||% rep(NA_character_, length(leaves))
  )
  phantoms <- lapply(compiled$phantom_specs, function(spec) {
    pjrt::pjrt_empty(dtype = spec$dtype, shape = spec$shape, device = device)
  })
  unname(c(consts, inputs, phantoms))
}
