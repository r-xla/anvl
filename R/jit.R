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
#' @param device (`NULL` | `character(1)` | [`nv_device`] | [`device_arg()`])\cr
#'   Target device, of the backend in force. When a concrete device is
#'   specified, all arrays are moved to it. `device_arg("<arg>")` reads the
#'   device from the named (static) argument at call time, for functions
#'   without array inputs such as constructors.
#'
#'   The default (`NULL`) infers the device at call time from the array inputs,
#'   falling back to [`default_device()`].
#'
#' @param ... Backend-specific options. See the **PJRT JIT arguments** and
#'   **Quickr JIT arguments** sections below for the options each backend
#'   accepts. An option no backend takes is rejected here; one that only
#'   another backend takes is rejected when the function is called on a backend
#'   that does not, since the backend is not known until then.
#' @inheritSection AnvlBackendPjrt PJRT JIT arguments
#' @inheritSection AnvlBackendQuickr Quickr JIT arguments
#'
#' @section Backend and device:
#' A jitted function runs on the backend in force *when it is called*
#' ([`default_backend()`], set with [`with_backend()`] / [`local_backend()`]),
#' so one `JitFunction` serves every backend, and a function created under one
#' backend and called under another runs on the latter. Array inputs must
#' belong to that backend; an array of another backend is rejected. Each
#' backend keeps its own compilation cache.
#'
#' The device is a choice within that backend. Setting `device` explicitly
#' enforces that the function always uses it, e.g. `"cuda:0"`, and copies every
#' array input to it. With `device = NULL` (default) the device is inferred from
#' the input arrays and the constants within the program; conflicting devices
#' are an error, and with no array to read a device from the default device is
#' used. A function without array inputs can read its device from a static
#' argument with `device = device_arg("<argname>")`.
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
#' # The same function runs on whichever backend is in force when it is called
#' with_backend("quickr", f(nv_array(1), nv_array(2)))
jit <- function(
  f,
  static = character(),
  cache_size = 100L,
  device = NULL,
  ...
) {
  static <- resolve_arg_names(f, static, "static")
  if (is_device_arg(device)) {
    assert_subset(device$argname, formalArgs2(f))
    # the device argument is always static
    static <- unique(c(static, device$argname))
  } else if (!is.null(device) && !is.character(device) && !is_device(device)) {
    cli_abort("{.arg device} must be a device, a device name, or {.fn device_arg}.")
  }
  assert_subset(static, formalArgs2(f))
  check_jit_options(names(list(...)))

  # One implementation per backend, created on first use. The backend is read
  # off the option on every call, so a function created under one backend runs
  # on another when that is the one in force -- and each implementation's
  # dispatcher validates that the array inputs belong to it.
  #
  # The wrapper takes `f`'s formals, so everything it closes over is named with
  # a `.jit_` prefix: a formal of `f` called `device` or `static` must not
  # shadow the configuration (prim_fill() has a `device` formal).
  .jit_cfg <- list(f = f, static = static, cache_size = cache_size, device = device, dots = list(...))
  .jit_fns <- list()
  .jit_runs <- list()

  wrapper <- function() {
    # Inside tracing: pass through to unwrapped function
    if (currently_tracing()) {
      .jit_cl <- match.call()
      .jit_cl[[1L]] <- .jit_cfg$f
      return(eval.parent(.jit_cl))
    }
    .jit_args <- lapply(as.list(match.call())[-1L], eval, envir = parent.frame())
    .jit_be <- default_backend()
    .jit_run <- .jit_runs[[.jit_be]]
    if (is.null(.jit_run)) {
      if (is.null(.jit_fns[[.jit_be]])) {
        .jit_fns[[.jit_be]] <<- do.call(
          jit_with_backend,
          c(
            list(
              f = .jit_cfg$f,
              static = .jit_cfg$static,
              cache_size = .jit_cfg$cache_size,
              backend = .jit_be,
              device = .jit_cfg$device
            ),
            .jit_cfg$dots
          )
        )
      }
      .jit_run <- attr(.jit_fns[[.jit_be]], "jit_run_args")
      if (is.null(.jit_run)) {
        # backend without a fast entry: call the JitFunction the generic way
        .jit_run <- function(args) do.call(.jit_fns[[.jit_be]], args)
      }
      .jit_runs[[.jit_be]] <<- .jit_run
    }
    # The args are already evaluated; the fast entry skips the inner
    # closure's match.call() + eval() re-capture (and do.call()).
    .jit_run(.jit_args)
  }
  formals(wrapper) <- formals2(f)
  class(wrapper) <- "JitFunction"
  wrapper
}

# The options a backend's `jit` method takes beyond the ones every backend
# gets, i.e. what may reach it through `jit()`'s `...`.
backend_jit_options <- function(backend) {
  setdiff(formalArgs2(globals$backends[[backend]]$jit), c("f", "static", "cache_size", "device", "..."))
}

# Reject what no backend could ever accept, at construction: a typo, or the
# `backend` argument this function used to have. An option only *another*
# backend takes is left for jit_with_backend(), since the backend a jitted
# function runs on is not known until it is called.
check_jit_options <- function(names) {
  if ("backend" %in% names) {
    cli_abort(c(
      "{.fn jit} has no {.arg backend} argument.",
      i = "A jitted function runs on the backend in force when it is called.",
      i = "Set it with {.fn with_backend} or {.fn local_backend}."
    ))
  }
  known <- unlist(lapply(names(globals$backends), backend_jit_options))
  unknown <- setdiff(names, unique(known))
  if (length(unknown)) {
    cli_abort(c(
      "No backend takes the {.arg {unknown}} {cli::qty(unknown)}option{?s}.",
      i = "The backend-specific options are {.arg {sort(unique(known))}}."
    ))
  }
  invisible(NULL)
}

# The implementation of `f` for one backend: its `jit` method's result, with
# `f`'s formals. `device` is NULL, a device or device name of that backend, or a
# device_arg().
jit_with_backend <- function(f, static, cache_size, backend, ...) {
  assert_backend(backend)
  unsupported <- setdiff(names(list(...)), c("device", backend_jit_options(backend)))
  if (length(unsupported)) {
    cli_abort(c(
      "The {.val {backend}} backend does not support the {.arg {unsupported}} {cli::qty(unsupported)}option{?s}.",
      i = "A jitted function runs on the backend in force when it is called, so a backend-specific option is
           only rejected once the function is called on a backend that does not take it.",
      i = "{.val {backend}} takes {.arg {backend_jit_options(backend)}}."
    ))
  }
  f_jit <- globals$backends[[backend]]$jit(f, static, cache_size, ...)
  # setting formals() rebuilds the function, so pick up the fast entry first
  run <- attr(f_jit, "jit_run_args")
  formals(f_jit) <- formals2(f)
  class(f_jit) <- "JitFunction"
  attr(f_jit, "jit_run_args") <- run
  f_jit
}

#' @title Select JIT device from a function argument
#' @description
#' Pass the result to [`jit()`]'s `device` argument to indicate that the
#' device should be read from a formal argument of the function being
#' compiled. At call time, the value of that argument -- a device of the
#' backend in force, or a device name -- is the device the program is compiled
#' for. The argument is static.
#'
#' This is intended for functions that have no dynamic array inputs from which
#' the device could otherwise be inferred (e.g. array constructors like
#' [prim_fill()] or [prim_iota()]).
#'
#' @param argname (`character(1)`)\cr
#'   Name of a formal argument of the function passed to [`jit()`].
#' @return (`AnvlDeviceArg`)\cr
#'   An object recognized by [`jit()`].
#' @seealso [`jit()`], [`nv_device()`]
#' @export
#' @examplesIf pjrt::plugins_downloaded("cpu")
#' f <- function(x) nv_scalar(1, device = x)
#' g <- jit(f, device = device_arg("x"))
#' g(nv_device("cpu"))
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
