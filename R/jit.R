#' @include backend.R
#' @include device.R
#' @include array.R
#' @title JIT compile a function
#' @description
#' Wraps a function so that it is traced and compiled on first call. Subsequent
#' calls with the same input structure, shapes, and dtypes hit an LRU cache and
#' skip recompilation.
#'
#' @param f (`function`)\cr
#'   Function to compile. Must accept and return [`AnvlArray`]s (and/or
#'   static arguments).
#' @param static (`character()` | `integer()`)\cr
#'   Names or positions of parameters of `f` that are *not* arrays. Static values are
#'   embedded as constants in the compiled program; a new compilation is triggered whenever
#'   a static value changes. For example useful when you want R control flow in your function.
#'
#'   Note that the values that are passed to static arguments must not have reference semantics.
#'   Such a value can be mutated in place while the cache key stays equal, which
#'   would silently reuse a program compiled from its old contents.
#'   One exception are closures, but there you need to ensure that their
#'   enclosing environment does not change in a way that modifies their behavior.
#'
#' @param cache_size (`integer(1)`)\cr
#'   Maximum number of compiled executables to keep in the LRU cache.
#' @param backend (`NULL` |  `character(1)`)\cr
#'   Compilation backend (e.g. `"pjrt"`, `"quickr"`).
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
#'   by the selected backend raises an error. See the **PJRT JIT arguments** and
#'   **Quickr JIT arguments** sections below for the options accepted by each
#'   backend.
#' @inheritSection AnvlBackendPjrt PJRT JIT arguments
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
#' @seealso
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
  assert_backend(backend)
  assert_subset(static, formalArgs2(f))

  f_jit <- globals$backends[[backend]]$jit(f, static, cache_size, ...)
  # setting formals() rebuilds the function, so pick up the fast entry first
  run <- attr(f_jit, "jit_run_args")
  formals(f_jit) <- formals2(f)
  class(f_jit) <- "JitFunction"
  attr(f_jit, "backend") <- backend
  attr(f_jit, "jit_run_args") <- run
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
#' g(nv_device("cpu", "pjrt"))
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
  # Lazily create per-backend jit functions (+ their evaluated-args fast entry)
  jit_fns <- list()
  jit_runs <- list()
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
    run <- jit_runs[[be]]
    if (is.null(run)) {
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
      run <- attr(jit_fns[[be]], "jit_run_args")
      if (is.null(run)) {
        # backend without a fast entry: call the JitFunction the generic way
        run <- function(args) do.call(jit_fns[[be]], args)
      }
      jit_runs[[be]] <<- run
    }
    # The args are already evaluated; the fast entry skips the inner
    # closure's match.call() + eval() re-capture (and do.call()).
    run(args)
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
      for (el in x) {
        scan(el)
      }
    }
  }
  if (length(static) == 0L) {
    for (a in args) {
      scan(a)
    }
  } else {
    nm <- rlang::names2(args)
    for (i in seq_along(args)) {
      if (!(nm[[i]] %in% static)) scan(args[[i]])
    }
  }
  if (is.na(found)) default_backend() else found
}


# The flat argument list a compile callback traces with, built from the `info`
# pjrt's dispatcher hands it: a static leaf traces as its value, a dynamic one
# as the aval the dispatcher already derived. There is deliberately no
# classification here -- the kind, dtype and shape below are the ones the cache
# key was built from, so the program we compile cannot disagree with the key it
# is filed under.
avals_from_dispatch <- function(info) {
  .mapply(
    function(leaf, is_static, av) {
      if (is_static) {
        return(leaf)
      }
      if (av$kind == "rdata") {
        # Bare R data: it has no dtype of its own, and the program decides what
        # it is uploaded as (see `RData`). Only the leaf's R type is read,
        # never its value -- the dispatcher keyed this entry on the type and
        # the shape, so a program that looked at the value would be served back
        # for a different one.
        return(nv_aval(typeof(leaf), av$shape))
      }
      nv_aval(as_dtype(av$dtype), av$shape)
    },
    list(info$leaves, info$is_static, info$avals),
    NULL
  )
}

# Reject static arguments with reference semantics, before their values are
# traced into a program.
#
# A static value is part of the executable-cache key: the dispatcher keeps the
# value and compares later calls against it with identical(). That is only
# sound while the value's content cannot change behind its identity. An
# environment or an external pointer can be mutated in place, leaving the key
# equal to one whose program was compiled from different contents -- a silently
# stale result rather than an error. So they are rejected here, at the first
# use of the value (the cache miss that traces it).
#
# Called with the call's argument list, which `match.call()` has named, and the
# jit's static argument names.
check_static_args <- function(args, static) {
  for (nm in intersect(rlang::names2(args), static)) {
    check_static_value(args[[nm]], nm)
  }
  invisible(NULL)
}

# Walk one static value, erroring on the first reference-semantics part of it.
# `path` is how that part is spelled from the argument, e.g. `opts$env`.
#
# Lists are walked because pjrt flattens a static list into one key leaf per
# element, so an environment inside one is keyed -- and goes stale -- exactly
# like a bare one. Attributes are not walked: they carry metadata rather than
# values the trace reads, and a formula's `.Environment` would make every
# static formula an error.
check_static_value <- function(x, path) {
  # The one reference-like static anvl passes itself (`jit(device = ...)`,
  # `device_arg()`): a device is an immutable, interned handle, so its identity
  # is its value.
  if (is_device(x)) {
    return(invisible(NULL))
  }
  kind <- reference_kind(x)
  if (!is.null(kind)) {
    cli_abort(
      c(
        "Static argument {.code {path}} has reference semantics: it is {kind}.",
        x = "A static value is part of the compilation cache key and is compared by identity,
             so mutating it in place would silently reuse a program compiled from its old contents.",
        i = "Pass it as a regular argument, or extract the plain values you need from it."
      ),
      call = NULL
    )
  }
  if (typeof(x) == "list") {
    nms <- rlang::names2(x)
    # .subset2() rather than `[[`: a classed static list must not get to decide
    # via a method of its own what its elements are.
    for (i in seq_along(x)) {
      check_static_value(.subset2(x, i), static_path(path, nms[[i]], i))
    }
  }
  invisible(NULL)
}

# A human description of `x`'s reference semantics, or NULL if it has none.
# Only R's own reference types are detected: a value-semantics object that some
# package mutates in place through C (a data.table, say) is indistinguishable
# from a plain list here.
reference_kind <- function(x) {
  what <- if (is.environment(x)) {
    "an environment"
  } else if (typeof(x) == "externalptr") {
    "an external pointer"
  } else {
    return(NULL)
  }
  if (is.object(x)) {
    sprintf("an object of class <%s> (%s)", class(x)[[1L]], what)
  } else {
    what
  }
}

static_path <- function(path, name, i) {
  if (!nzchar(name)) {
    sprintf("%s[[%i]]", path, i)
  } else if (identical(make.names(name), name)) {
    sprintf("%s$%s", path, name)
  } else {
    sprintf("%s[[\"%s\"]]", path, name)
  }
}

# The devices of the call's array inputs, for compile_pjrt()'s device inference.
# pjrt has already checked they agree; this only converts them to anvl devices.
dispatch_arg_devices <- function(info) {
  is_array <- !info$is_static & vapply(info$leaves, is_anvl_array, logical(1))
  lapply(info$leaves[is_array], tengen::device)
}
