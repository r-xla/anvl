as_default_dtypes <- function(dtypes) {
  vapply(names(dtypes), function(category) as.character(as_dtype(dtypes[[category]])), character(1L))
}

resolve_option_default_dtypes <- function(value, backend) {
  names_ <- names(value)
  override <- as_default_dtypes(value[names_ %in% c("float", "int")])
  named <- if (backend %in% names_) as_default_dtypes(value[[backend]]) else character()
  override[names(named)] <- named
  override
}

option_default_dtypes <- function(backend) {
  value <- getOption("anvl.default_dtypes")
  if (is.null(value)) {
    return(character())
  }
  resolve_option_default_dtypes(value, backend)
}

apply_default_dtypes <- function(baseline, override) {
  for (category in names(override)) {
    baseline[[category]] <- as_dtype(override[[category]])
  }
  baseline
}

# The effective pair for `backend`: the override over its registered defaults.
effective_default_dtypes <- function(backend) {
  apply_default_dtypes(registered_default_dtypes(backend), option_default_dtypes(backend))
}

#' @title Default Data Types
#' @description
#' The default data types for the [active backend][active_backend()].
#' They decide the data type an R value is materialized at when it cannot be
#' inferred from another operand.
#'
#' This includes array creation via (`nv_array(1)`) or passing R values to unary functions
#' (`prim_exp(1)`).
#'
#' `default_dtypes()` reports both categories at once; `default_float()` and
#' `default_int()` report one each.
#'
#' Each backend registers its own -- `f32` / `i32` for `"pjrt"`, `f64` / `i32`
#' for `"quickr"` -- and the `anvl.default_dtypes` option overrides them.
#' [`local_default_dtypes()`] and [`with_default_dtypes()`] set the option for
#' a scope.
#'
#' Below, we configure any backend to use the default `f64` for floats and `i64` for integers:
#'
#' ```
#' options(anvl.default_dtypes = c(float = "f64", int = "i64"))
#' ```
#'
#' You can also only change the float dtype, leaving
#' the backend's default integer dtype unchanged:
#'
#' ```
#' options(anvl.default_dtypes = c(float = "f64"))
#' ```
#'
#' It is also possible to specify the defaults per-backend:
#'
#' ```
#' options(anvl.default_dtypes = list(
#'   pjrt = list(float = "f64", int = "i64"),
#'   quickr = list(int = "i32")
#' ))
#' ```
#'
#' An entry that names a backend wins over the categories beside it.
#'
#' The defaults decide only what a value becomes when *nothing else does*: an R
#' value that meets a typed array of its own category still takes that array's
#' data type, whatever the default (`vignette("type-promotion")`). The data
#' type you name is taken on trust, so one that does not fit is an error where
#' the data is allocated or the program compiled rather than where it is set.
#' Which ones fit is the backend's own business -- see the *Supported data
#' types* section of [`AnvlBackendPjrt()`] and of [`AnvlBackendQuickr()`],
#' which has only `f64`, `i32` and `bool`, so both `"f32"` and `"i64"` are
#' errors there. A compiled program is keyed on the defaults it was compiled
#' under, so changing them never serves a stale program.
#'
#' @return `default_dtypes()` returns a named `list` with elements `float` and
#'   `int`, each a [`DataType`]
#' @seealso [`local_default_dtypes()`], [`with_default_dtypes()`],
#'   [`with_dtypes()`]
#' @examplesIf pjrt::plugins_downloaded()
#' with_backend("quickr", default_dtypes())
#' with_backend("pjrt", default_dtypes())
#' default_dtypes()
#' default_float()
#' default_int()
#' dtype(nv_array(1.5))
#' @export
default_dtypes <- function() {
  current_default_dtypes()
}

#' @rdname default_dtypes
#' @export
default_float <- function() {
  current_default_dtypes()$float
}

#' @rdname default_dtypes
#' @export
default_int <- function() {
  current_default_dtypes()$int
}

registered_default_dtypes <- function(backend) {
  registered <- globals$backends[[backend]]$default_dtypes
  if (is.null(registered)) {
    cli_abort("The {.val {backend}} backend has no default data types.")
  }
  registered
}

# The dispatcher's `context` resolver for `backend`: a function returning the
# current defaults as a character vector, which keys a compiled program on the
# defaults it was compiled under. Called on every dispatch, so everything that
# does not change per call is resolved once, here.
default_dtypes_context <- function(backend) {
  registered <- registered_default_dtypes(backend)
  fallback <- c(float = as.character(registered$float), int = as.character(registered$int))
  function() {
    value <- getOption("anvl.default_dtypes")
    if (is.null(value)) {
      return(fallback)
    }
    override <- resolve_option_default_dtypes(value, backend)
    pair <- fallback
    pair[names(override)] <- override
    pair
  }
}

# The inverse, for the compile callback: the defaults a trace is pinned to,
# from the context the dispatcher keyed the entry on. `NULL` stays `NULL`.
default_dtypes_from_key <- function(key) {
  if (is.null(key)) {
    return(NULL)
  }
  list(float = as_dtype(key[["float"]]), int = as_dtype(key[["int"]]))
}

# Used for context managers: insert dtypes into default default_dtypes
merged_default_dtypes <- function(dtypes, backend) {
  # A program is compiled for one backend, and a trace reads the entry of that
  # backend, so an override filed under another could not reach it. Naming one
  # explicitly is a mistake worth reporting rather than a silent no-op.
  desc <- globals[["CURRENT_DESCRIPTOR"]]
  if (!is.null(backend) && !is.null(desc) && !identical(backend, desc$backend)) {
    cli_abort(c(
      "Cannot set the default data types of the {.val {backend}} backend here.",
      i = "This trace is compiled for the {.val {desc$backend}} backend, which is
           the only one an override in it can reach."
    ))
  }
  backend <- backend %||% active_backend()
  new <- as_default_dtypes(dtypes)
  current <- getOption("anvl.default_dtypes")
  if (is.null(current)) {
    current <- list()
  } else {
    # Read what is there, so that merging into a value nothing recognizes fails
    # here rather than on the next read.
    resolve_option_default_dtypes(current, backend)
    current <- as.list(current)
  }
  entry <- if (is.null(current[[backend]])) {
    character()
  } else {
    as_default_dtypes(current[[backend]])
  }
  entry[names(new)] <- new
  current[[backend]] <- entry
  list(anvl.default_dtypes = current)
}

#' @title Set the Default Data Types
#' @description
#' Set the default data types (see [`default_dtypes()`]) for a scope:
#' `local_default_dtypes()` until the calling frame exits,
#' `with_default_dtypes()` for the duration of `code`. Both write the
#' `anvl.default_dtypes` option for one backend, and change only the
#' categories they name.
#'
#' @details
#' Inside a [`jit()`]ted body the defaults the program was keyed on are the
#' *baseline* and an override applies to its scope, so one program can use
#' different precisions in different parts of itself. Only that baseline is
#' part of the compilation cache key, so **an override in a body must not
#' change between calls: write it out literally rather than reading it from a
#' variable.** `with_default_dtypes(c(float = prec), ...)` with a `prec` that
#' later changes keeps serving the program traced at the first value, exactly
#' as a changing `dtype` argument would -- and just as silently. Nothing
#' checks this for you.
#'
#' @param dtypes (named `character()` | named `list()`)\cr
#'   A mapping of the data type categories (`float` and `int`) to data types,
#'   e.g. `c(float = "f64", int = "i32")`. Each may be a string or a
#'   [`DataType`]. Can also be a partial override, such as `c(float = "f64")`,
#'   in which case the category it does not name is left as it is.
#' @param backend (`NULL` | `character(1)`)\cr
#'   The backend whose defaults to set. Uses [`active_backend()`] by default.
#' @param envir (`environment`)\cr
#'   The environment to scope the change to.
#' @param code An expression to evaluate with the given defaults.
#' @return `local_default_dtypes()` returns the previous values of the options
#'   it set, invisibly. `with_default_dtypes()` returns the result of
#'   evaluating `code`.
#' @seealso [`default_dtypes()`], [`local_backend()`], [`with_backend()`]
#' @examplesIf pjrt::plugins_downloaded()
#' with_default_dtypes(c(float = "f64"), dtype(nv_array(1.5)))
#' # A value that meets a typed array still takes that array's data type
#' with_default_dtypes(c(float = "f64"), dtype(nv_array(1, dtype = "f32") + 1.5))
#' # Untyped values in one program can commit at different precisions
#' jit(function() {
#'   list(single = nv_fill(0, 2), double = with_default_dtypes(c(float = "f64"), nv_fill(0, 2)))
#' })()
#' @export
local_default_dtypes <- function(dtypes, backend = NULL, envir = parent.frame()) {
  withr::local_options(merged_default_dtypes(dtypes, backend), .local_envir = envir)
}

#' @rdname local_default_dtypes
#' @export
with_default_dtypes <- function(dtypes, code, backend = NULL) {
  withr::with_options(merged_default_dtypes(dtypes, backend), code)
}

#' @title Run a Function at Given Data Types
#' @description
#' `with_dtypes()` wraps `f` into a function that works at the data types
#' `dtypes` names: on each call every array argument of a category `dtypes`
#' names is converted to that data type, the defaults (see
#' [`default_dtypes()`]) are set to the `float` / `int` entries for the
#' duration of the call, and every returned array of a named category is
#' converted as well.
#'
#' ```r
#' nv_add_f64 <- with_dtypes(nv_add, c(float = "f64"))
#' ```
#'
#' A category `dtypes` does not name is left alone, in the arguments, in the
#' body and in the result.
#'
#' @details
#' Note that `f` itself can also change the default data types, which overrides the
#' defaults configured by `with_dtypes()`.
#' @param f (`function`)\cr
#'   The function to wrap.
#' @param dtypes (named `character()` | named `list()`)\cr
#'   A mapping of the data type categories (`float`, `int` and `uint`) to data
#'   types, e.g. `c(float = "f64", int = "i64")`. Each may be a string or a
#'   [`DataType`]. A category it does not name is left as it is.
#' @return A `function` with the same arguments as `f`, a `JitFunction` if `f`
#'   was one.
#' @seealso [`default_dtypes()`], [`local_default_dtypes()`]
#' @examplesIf pjrt::plugins_downloaded()
#' add_f64 <- with_dtypes(nv_add, c(float = "f64"))
#' # An `f32` argument is converted, and the result comes back as `f64`
#' dtype(add_f64(nv_array(1, dtype = "f32"), 2.5))
#' # A category that is not named is untouched
#' dtype(add_f64(nv_array(1L, dtype = "i32"), 2L))
#' # `uint` is converted too, but sets no default
#' dtype(with_dtypes(nv_add, c(uint = "ui32"))(nv_array(1L, dtype = "ui8"), 2L))
#' @export
with_dtypes <- function(f, dtypes) {
  if (!is.function(f)) {
    cli_abort("{.arg f} must be a function, not {.obj_type_friendly {f}}.")
  }
  .dtypes_targets <- assert_dtype_categories(dtypes)
  .dtypes_defaults <- dtypes[names(dtypes) %in% c("float", "int")]

  cfg <- jit_config(f)
  .dtypes_f <- cfg$f %||% f

  wrapper <- if (length(.dtypes_defaults)) {
    function() {
      local_default_dtypes(.dtypes_defaults)
      .dtypes_args <- lapply(as.list(match.call())[-1L], eval, envir = parent.frame())
      convert_call(.dtypes_f, .dtypes_args, .dtypes_targets)
    }
  } else {
    function() {
      .dtypes_args <- lapply(as.list(match.call())[-1L], eval, envir = parent.frame())
      convert_call(.dtypes_f, .dtypes_args, .dtypes_targets)
    }
  }
  formals(wrapper) <- formals2(.dtypes_f)

  if (is.null(cfg)) {
    return(wrapper)
  }
  do.call(
    jit,
    c(
      list(
        f = wrapper,
        static = cfg$static,
        cache_size = cfg$cache_size,
        device = cfg$device
      ),
      cfg$dots
    )
  )
}

convert_call <- function(f, args, targets) {
  args <- convert_tree(args, targets)
  convert_tree(do.call(f, args), targets)
}

assert_dtype_categories <- function(dtypes) {
  categories <- names(dtypes)
  ok <- length(dtypes) &&
    !is.null(categories) &&
    !anyDuplicated(categories) &&
    all(categories %in% c("float", "int", "uint"))
  if (!ok) {
    cli_abort(c(
      "{.arg dtypes} must map the data type categories to data types.",
      i = "The categories are {.val float}, {.val int} and {.val uint},
           e.g. {.code c(float = \"f64\")}."
    ))
  }
  lapply(dtypes, as_dtype)
}

convert_tree <- function(x, targets) {
  map_tree(x, convert_leaf, targets = targets)
}

convert_leaf <- function(x, targets) {
  if (!is_arrayish(x, convert_ok = FALSE)) {
    return(x)
  }
  category <- convertible_dtype_category(peek_dtype(x))
  if (is.null(category) || !category %in% names(targets)) {
    return(x)
  }
  nv_convert(x, targets[[category]])
}

convertible_dtype_category <- function(dtype) {
  if (is_dtype_float(dtype)) {
    "float"
  } else if (is_dtype_int(dtype)) {
    "int"
  } else if (is_dtype_uint(dtype)) {
    "uint"
  } else {
    NULL
  }
}

# The default dtypes (see `default_dtypes()`) in force here.
current_default_dtypes <- function() {
  desc <- globals[["CURRENT_DESCRIPTOR"]]
  if (is.null(desc)) {
    # Eager mode: The active backend's default_dtypes with a possible
    # overwrite by the anvl.default_dtypes option
    return(effective_default_dtypes(active_backend()))
  }
  # Insert options over desc$default_dtypes -- the pair the dispatcher keyed
  # the program on.
  apply_default_dtypes(desc$default_dtypes, option_default_dtypes(desc$backend))
}

default_dtype <- function(x, defaults = current_default_dtypes()) {
  if (!is.numeric(x) && !is.logical(x)) {
    cli_abort("No default type for {.obj_type_friendly {x}}.")
  }
  default_dtype_r(typeof(x), defaults)
}

resolve_default_dtype <- function(data, dtype, defaults = current_default_dtypes()) {
  if (is.null(dtype) && (is.numeric(data) || is.logical(data))) {
    return(default_dtype(data, defaults))
  }
  dtype
}

default_dtype_r <- function(r_type, defaults = current_default_dtypes()) {
  switch(
    r_type,
    double = defaults$float,
    integer = defaults$int,
    logical = as_dtype("bool"),
    cli_abort("No default type for R type {.val {r_type}}")
  )
}
