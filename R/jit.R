#' @include backend.R
#' @include device.R
#' @include array.R
#' @title JIT compile a function
#' @description
#' Wraps a function so that it is traced and compiled on first call. Subsequent
#' calls with the same input structure, shapes, and dtypes hit an LRU cache and
#' skip recompilation. Unlike [`xla()`], the compiled executable is not created
#' eagerly but lazily on the first invocation.
#'
#' @param f (`function`)\cr
#'   Function to compile. Must accept and return [`AnvlArray`]s (and/or
#'   static arguments).
#' @param static (`character()` | `integer()`)\cr
#'   Names or positions of parameters of `f` that are *not* arrays. Static values are
#'   embedded as constants in the compiled program; a new compilation is triggered whenever
#'   a static value changes. For example useful when you want R control flow in your function.
#' @param cache_size (`integer(1)`)\cr
#'   Maximum number of compiled executables to keep in the LRU cache.
#' @param backend (`NULL` |  `character(1)`)\cr
#'   Compilation backend (e.g. `"xla"`, `"quickr"`).
#'   The special value `"auto"` defers backend selection to call-time.
#'   `NULL` (default) respects `device` and otherwise falls back to [`default_backend()`].
#' @param device (`NULL` | `character(1)` | [`nv_device`] | `device_arg()`)\cr
#'   Target device. When a concrete device is specified, all arrays
#'   are moved to it.
#'
#'   The default (`NULL`) infers the device at call time,
#'   falling back to [`default_device()`].
#'
#'   In order to use dynamic device selection with the `"auto"` backend (e.g. for functions without
#'   dynamic inputs such as constant creation), set `device = device_arg("<arg>")`.
#'
#' @param ... Backend-specific options. Passing an option that is not supported
#'   by the selected backend raises an error. See the **XLA JIT arguments** and
#'   **Quickr JIT arguments** sections below for the options accepted by each
#'   backend.
#' @inheritSection AnvlBackendXla XLA JIT arguments
#' @inheritSection AnvlBackendQuickr Quickr JIT arguments
#'
#' @section Device and Backend selection:
#' There are various ways to specify which device and which backend to use.
#'
#' **Concrete backend**:
#' In the case where we fix a concrete backend (backend is not `"auto"`), the device can be
#' inferred or set explicitly.
#' Setting the device explicitly allows you to enforce that the function always uses the specified
#' device, e.g. `"cuda:0"`.
#' If the `device` argument is set, all encountered arrays are copied to it.
#'
#' If the device is not specified (`NULL`; default) the device will be inferred from the input
#' arrays and the constants within the program. If conflicting devices are found, an error
#' is thrown. If no array with a device is found, we fall back to the default device.
#'
#' **Auto backend**:
#' When setting `backend = "auto"`, the backend will be inferred from the array inputs and
#' otherwise fall back to the default backend.
#' If you want to `jit()` a function without array inputs but make it work with different devices,
#' set `device = device_arg("<argname>")` where `<argname>` is the name of the argument specifying
#' the device. Note that this is only necessary with the `"auto"` backend.
#' When using a concrete backend, you can just specify the device via a static argument.
#'
#' @section Jitting in a Package:
#' To `jit()` a function defined in an R package, prefer the `@jit` roxygen
#' tag over a top-level `jit()` call:
#'
#' ```r
#' #' @export
#' #' @jit static = c("flag")
#' my_fun <- function(x, flag) if (flag) x + 1 else x * 2
#' ```
#'
#' This delegates the wrapping to [`jit_roclet()`], which records the
#' tagged functions in `R/jit-registry.R`. The wrapping itself happens at
#' package build time via [`apply_jit_registry()`] in `R/zzz.R`, so the
#' resulting `JitFunction` is byte-compiled with the rest of the package
#' instead of being rebuilt on every `.onLoad`.
#'
#' See [`jit_roclet()`] for the one-time setup of the roclet in your
#' package.
#'
#' @return A `JitFunction` (a `function` with the same formals as `f`).
#'   The returned wrapper expects [`AnvlArray`] inputs and returns
#'   [`AnvlArray`] values.
#' @seealso [`xla()`] for ahead-of-time compilation, [`jit_eval()`] for evaluating an expression once,
#'   [`jit_roclet()`] for the `@jit` tag used inside R packages.
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' f <- jit(function(x, y) x + y)
#' f(nv_array(1), nv_array(2))
#'
#' # Static arguments enable data-dependent control flow
#' g <- jit(function(x, flag) {
#'   if (flag) x + 1 else x * 2
#' }, static = "flag")
#' g(nv_array(3), TRUE)
#' g(nv_array(3), FALSE)
#'
#' @examplesIf requireNamespace("quickr", quietly = TRUE)
#' with_backend("quickr", {
#'   h <- jit(function(x, y) x + y)
#'   h(nv_array(1), nv_array(2))
#' })
jit <- function(
  f,
  static = character(),
  cache_size = 100L,
  backend = NULL,
  device = NULL,
  ...
) {
  static <- resolve_arg_names(f, static, "static")
  if (is_device_arg(device)) {
    if (!(device$argname %in% static)) {
      static <- c(static, device$argname)
    }
    if (is.null(backend) || identical(backend, "auto")) {
      return(jit_auto(f, static, cache_size, device_argname = device$argname, ...))
    }
    # There is really no need to support this. device_arg() is really about being able to detect
    # backend at the start so we know which backend's jit method to call.
    cli_abort(c(
      "device = device_arg() is only allowed with backend `NULL` or \"auto\".",
      i = "Just use a static argument for the device selection"
    ))
  }
  if (identical(backend, "auto")) {
    if (is_device(device)) {
      cli_abort("Don't provide a concrete device when using the \"auto\" backend.")
    } else {
      return(jit_auto(f, static, cache_size, device = device, ...))
    }
  }
  # device might still be NULL, which means infer from encountered arrays
  resolved <- resolve_device(device, backend)
  device <- resolved[[1L]]
  backend <- resolved[[2L]]

  jit_with_backend(f, static, cache_size, backend, device = device, ...)
}

jit_with_backend <- function(f, static, cache_size, backend, ...) {
  cache <- xlamisc::LRUCache$new(cache_size)
  assert_backend(backend)
  assert_subset(static, formalArgs2(f))

  f_jit <- globals$backends[[backend]]$jit(f, static, cache, ...)
  formals(f_jit) <- formals2(f)
  class(f_jit) <- "JitFunction"
  attr(f_jit, "backend") <- backend
  f_jit
}

#' @title Select JIT device from a function argument
#' @description
#' Pass the result to [`jit()`]'s `device` argument to indicate that the
#' device should be read from a formal argument of the function being
#' compiled. At call time, the value of that argument is used to derive the
#' backend via [`backend()`] dispatch and is forwarded to the backend-specific
#' JIT as the compilation device.
#'
#' This is intended for functions that have no dynamic array inputs from which
#' the backend could otherwise be detected (e.g. array constructors like
#' [prim_fill()] or [prim_iota()]).
#'
#' @param argname (`character(1)`)\cr
#'   Name of a formal argument of the function passed to [`jit()`].
#' @return (`AnvlDeviceArg`)\cr
#'   An object recognized by [`jit()`].
#' @seealso [`jit()`], [`backend()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' f <- function(x) nv_scalar(1, device = x)
#' g <- jit(f, backend = "auto", device = device_arg("x"))
#' g(nv_device("cpu", "xla"))
device_arg <- function(argname) {
  assert_string(argname)
  structure(list(argname = argname), class = "AnvlDeviceArg")
}

# Translate a character-or-integer argument selector into character names
# of the formals of `f`. `arg` is used in error messages. Rejects `"..."`
# (whether supplied by name or resolved from an integer position), since it
# does not name a single argument.
resolve_arg_names <- function(f, x, arg) {
  if (is.null(x)) {
    return(x)
  }
  if (is.integer(x)) {
    nms <- formalArgs2(f)
    if (any(x < 1L | x > length(nms))) {
      cli_abort("{.arg {arg}} index out of range.")
    }
    x <- nms[x]
  }
  if ("..." %in% x) {
    cli_abort("{.arg {arg}} must not contain {.val ...}.")
  }
  x
}

#' @export
backend.JitFunction <- function(x, ...) {
  attr(x, "backend")
}

jit_auto <- function(f, static, cache_size, device = NULL, device_argname = NULL, ...) {
  # A concrete device would pin the backend and defeat the purpose of `auto`;
  # `jit()` collapses that case to `jit_with_backend` before we get here.
  if (is_device(device)) {
    cli_abort("Internal error: jit_auto called with a concrete device; backend should have been pinned.")
  }
  # Lazily create per-backend jit functions
  jit_fns <- list()
  dots <- list(...)
  if (!is.null(device_argname)) {
    assert_subset(device_argname, formalArgs2(f))
    # the device argument is always static
    static <- unique(c(static, device_argname))
  }

  wrapper <- function() {
    # Inside tracing: pass through to unwrapped function
    if (currently_tracing()) {
      cl <- match.call()
      cl[[1L]] <- f
      return(eval.parent(cl))
    }
    args <- lapply(as.list(match.call())[-1L], eval, envir = parent.frame())
    be <- if (!is.null(device_argname) && !is.null(args[[device_argname]])) {
      dev_val <- args[[device_argname]]
      if (is.character(dev_val)) default_backend() else backend(dev_val)
    } else {
      jit_auto_detect_backend(args, static)
    }
    if (is.null(jit_fns[[be]])) {
      jit_fns[[be]] <<- do.call(
        jit_with_backend,
        c(
          list(f = f, static = static, cache_size = cache_size, backend = be),
          if (!is.null(device_argname)) {
            list(device = device_arg(device_argname))
          } else if (!is.null(device)) {
            list(device = device)
          },
          dots
        )
      )
    }
    do.call(jit_fns[[be]], args)
  }
  formals(wrapper) <- formals2(f)
  class(wrapper) <- "JitFunction"
  attr(wrapper, "backend") <- "auto"
  wrapper
}

# Determine the backend from a call's (already evaluated) arguments: the single
# non-"plain" backend among the AnvlArray leaves, or default_backend() if none.
# A direct short-circuiting scan that reads `$backend` as a field -- this is on
# the hot eager-dispatch path, so it avoids flatten()/vapply()/unique()/`%in%`.
jit_auto_detect_backend <- function(args, static = character()) {
  found <- NA_character_
  scan <- function(x) {
    if (is_anvl_array(x)) {
      b <- x$backend
      if (!identical(b, "plain")) {
        if (is.na(found)) {
          found <<- b
        } else if (!identical(found, b)) {
          cli_abort(c(
            "Cannot auto-detect backend: inputs use multiple backends.",
            i = "Found backends: {.val {c(found, b)}}",
            i = "Pass {.code backend =} to {.fn jit} or convert inputs to a common backend."
          ))
        }
      }
    } else if (is.list(x) && !is.object(x)) {
      for (el in x) scan(el)
    }
  }
  if (length(static) == 0L) {
    for (a in args) scan(a)
  } else {
    nm <- rlang::names2(args)
    for (i in seq_along(args)) {
      if (!(nm[[i]] %in% static)) scan(args[[i]])
    }
  }
  if (is.na(found)) default_backend() else found
}

jit_prepare_call <- function(call, eval_env, static, device = NULL, backend) {
  args <- as.list(call)[-1L]
  args <- lapply(args, eval, envir = eval_env)
  jit_prepare_args(args, static, device = device, backend = backend)
}

# The structure-and-validation half of jit_prepare_call, operating on an
# already-evaluated argument list (so the native dispatcher's compile callback,
# which receives evaluated args, can reuse it).
jit_prepare_args <- function(args, static, device = NULL, backend) {
  assert_choice(backend, c("xla", "quickr"))

  # Fast path for a flat argument list -- no nested bare-list or NULL pytree
  # nodes, the common case for primitive calls. build_tree()/flatten() would
  # recursively walk the args to produce exactly these three values; building
  # them directly avoids that overhead on the hot dispatch path. Leaves
  # (AnvlArrays, atomics, ...) are either non-lists or classed objects, so the
  # structural-node test is `is.null() || (is.list() && !is.object())`.
  if (length(args) > 0L && !any(vapply(args, function(a) is.null(a) || (is.list(a) && !is.object(a)), logical(1)))) {
    args_flat <- unname(args)
    is_static_flat <- rlang::names2(args) %in% static
    in_tree <- ListNode(lapply(seq_along(args), LeafNode), names(args))
  } else {
    in_tree <- build_tree(mark_some(args, static))
    args_flat <- flatten(args)
    is_static_flat <- in_tree$marked
    in_tree$marked <- NULL
    class(in_tree) <- c("ListNode", "Node")
  }

  # Resolve device: device_arg() → extract from args, string/device → nv_device(),
  # NULL → infer from inputs, then fall back to PJRT_PLATFORM default.
  if (is_device_arg(device)) {
    device <- args[[device$argname]]
  }

  # Determine allocation device:
  # if device is specified -> use it
  # else, use first found device; if no device was found, leave it at NULL and don't check
  # device (will be inferred during tracing)
  allocation_device <- if (is.null(device)) {
    found_device <- NULL
    # If any input lives on a device, use it instead
    for (i in seq_along(args_flat)) {
      if (!is_static_flat[[i]] && is_anvl_array(args_flat[[i]])) {
        found_device <- device(args_flat[[i]])
        break
      }
    }
    found_device
  } else {
    device
  }

  # whether we are copying between devices. This happens when specific device was specified in jit()
  copy_to_device <- !is.null(device)

  .mapply(
    function(x, is_static, i) if (is_static) x else check_jit_input(x, allocation_device, in_tree, i, copy_to_device),
    list(args_flat, is_static_flat, seq_along(args_flat)),
    NULL
  )

  list(
    args = args,
    args_flat = args_flat,
    is_static_flat = is_static_flat,
    in_tree = in_tree,
    device = allocation_device
  )
}

to_avals <- function(args_flat, is_static_flat) {
  Map(
    function(x, is_static) {
      if (is_static) {
        x
      } else if (is_anvl_array(x)) {
        nv_aval(dtype(x), shape(x), ambiguous(x))
      } else if (is_valid_r_lit(x)) {
        nv_aval(default_dtype(x), integer(), ambiguous = TRUE)
      } else if (is_valid_r_array(x)) {
        nv_aval(default_dtype(x), as.integer(dim(x)), ambiguous = TRUE)
      } else {
        cli_abort("internal error: invalid input type for jit: {.cls {class(x)[1L]}}")
      }
    },
    args_flat,
    is_static_flat
  )
}

# Cheap executable-cache key material for a call's input signature. It encodes
# exactly the fields that distinguish a compilation -- per dynamic leaf the
# (dtype, shape, ambiguity), and for static leaves the value itself -- mirroring
# to_avals(), but as plain lists read straight from the cached array metadata
# instead of constructing nv_aval objects. This lets the dispatch probe the
# cache before building the (more expensive) abstract values, so a cache hit
# avoids to_avals() entirely; the avals are only needed to compile on a miss.
jit_key_leaves <- function(args_flat, is_static_flat) {
  .mapply(
    function(x, is_static) {
      if (is_static) {
        x
      } else if (is_anvl_array(x)) {
        list(x$dtype, x$shape, x$ambiguous)
      } else if (is_valid_r_lit(x)) {
        list(default_dtype(x), integer(), TRUE)
      } else if (is_valid_r_array(x)) {
        list(default_dtype(x), as.integer(dim(x)), TRUE)
      } else {
        cli_abort("internal error: invalid input type for jit: {.cls {class(x)[1L]}}")
      }
    },
    list(args_flat, is_static_flat),
    NULL
  )
}

# Check whether an input to jit is valid (w.r.t. information available before tracing)
# We don't convert yet as the concrete device is only known after tracing (respecting found constant's device)
# in_tree and i are only used for good error messages
check_jit_input <- function(x, alloc_device, in_tree = NULL, i = NULL, copy_to_device) {
  make_path <- function() {
    if (!is.null(in_tree) && !is.null(i)) tree_path(in_tree, i) else ""
  }
  if (is_anvl_array(x)) {
    # only single device currently
    if (backend(x) == "quickr") {
      return(x)
    }
    # there any input is valid as we will move it to it
    if (copy_to_device) {
      return(x)
    }

    # allocation device can be NULL if e.g. all inputs are R objects and no concrete device
    # was enforced in jit()
    if (!is.null(alloc_device) && !eq_device(device(x), alloc_device)) {
      # this can happen when there are multiple input devices but we are auto-detecting device
      path <- make_path()
      cli_abort(c(
        "Found AnvlArray input {.arg {path}} on unexpected device {device(x)}",
        i = "when using jit(f, device = NULL), ensure that all inputs live on the same device"
      ))
    }

    return(x)
  }

  if (is_valid_r(x)) {
    return(x)
  }
  path <- make_path()
  msg <- if (nzchar(path)) {
    "Attempted to autoconvert {.arg {path}} to an {.cls AnvlArray}."
  } else {
    "Attempted to autoconvert input to an {.cls AnvlArray}."
  }
  cli_abort(c(
    msg,
    i = "Expected an {.cls AnvlArray}, a length-1 atomic scalar, or an {.code is.array()} value.",
    x = "Got {.cls {class(x)[1]}} of length {length(x)}."
  ))
}

jit_wrap_outputs <- function(out_flat, out_tree, ambiguous_out, backend) {
  if (!is.null(ambiguous_out)) {
    out_flat <- Map(function(val, amb) nv_array(val, ambiguous = amb, backend = backend), out_flat, ambiguous_out)
  } else {
    out_flat <- lapply(out_flat, nv_array, backend = backend)
  }
  unflatten(out_tree, out_flat)
}


#' @title JIT-compile and evaluate an expression
#' @description
#' Convenience wrapper that JIT-compiles and immediately evaluates a single expression.
#' Equivalent to wrapping `expr` in an anonymous function, calling [`jit()`] on it, and
#' invoking the result.
#' Useful if you want to evaluate an expression once.
#' @param expr (NSE)\cr
#'   Expression to compile and evaluate.
#' @param ... Backend-specific options forwarded to [`jit()`] (e.g. `device`
#'   for the `"xla"` backend, `unwrap` for the `"quickr"` backend).
#' @return (`any`)\cr
#'   Result of the compiled and evaluated expression.
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(1, 2, 3), dtype = "f32")
#' jit_eval(x + x)
jit_eval <- function(expr, ...) {
  expr <- substitute(expr)
  eval_env <- new.env(parent = parent.frame())
  jit(\() eval(expr, envir = eval_env), ...)()
}
