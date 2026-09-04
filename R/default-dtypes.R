# The default data types: what an R double and an R integer commit to when
# nothing else decides one. Registered per backend, overridden by two global
# options, and pinned on a trace as the pair the dispatcher keyed the compiled
# program on.

# The dtypes a default may be set to, per category. A float default is limited
# to what pjrt's dispatcher can key and wrap and what `gradient()` supports; an
# integer default to the signed dtypes an R integer builds at directly (see
# `rdata_builds_directly()`): a narrower one would make the upload of an R
# argument go through R's coercion, which wraps where the program's `convert`
# clamps, and an unsigned one cannot hold a negative R integer at all.
default_dtype_choices <- list(
  float = c("f32", "f64"),
  int = c("i32", "i64")
)

# Validate one default; `what` names it in the error, as cli markup.
check_default_dtype <- function(x, category, what) {
  allowed <- default_dtype_choices[[category]]
  dtype <- tryCatch(as_dtype(x), error = function(e) NULL)
  if (is.null(dtype) || !(as.character(dtype) %in% allowed)) {
    cli_abort(
      c(
        paste0(what, " must be one of {.val {allowed}}."),
        x = "Got {.val {x}}."
      ),
      call = NULL
    )
  }
  dtype
}

default_dtype_option <- function(category) {
  sprintf("anvl.default_%s", category)
}

# The override option for one category, validated. `NULL` where it is unset,
# so a caller can fall back to whatever baseline applies to it.
option_default_dtype <- function(category) {
  option <- default_dtype_option(category)
  value <- getOption(option)
  if (is.null(value)) {
    return(NULL)
  }
  check_default_dtype(value, category, sprintf("Option {.code %s}", option))
}

# One category's effective default for `backend`: the override option over the
# backend's registered default.
default_dtype_for <- function(backend, category) {
  option_default_dtype(category) %||% registered_default_dtypes(backend)[[category]]
}

# The effective pair for `backend`. `default_dtypes()` is this for the backend
# in force; a dispatcher's context resolver asks it for its own backend.
effective_default_dtypes <- function(backend) {
  list(
    float = default_dtype_for(backend, "float"),
    int = default_dtype_for(backend, "int")
  )
}

#' Default data types
#'
#' An R value entering a program has no data type of its own (see
#' [`RData`]). Where nothing it meets decides one, it commits to a default
#' for its R type: a double to the default *float*, an integer to the default
#' *integer*, a logical to `bool`. These are the data types [`nv_array()`] and
#' [`nv_scalar()`] build at when `dtype` is not given, that a literal in a
#' [`jit()`]ted function commits to, and that [`nv_seq()`], [`nv_eye()`] and the
#' random samplers use when their `dtype` is `NULL`.
#'
#' Each backend registers its own defaults -- `f32` / `i32` for `"pjrt"`, `f64`
#' / `i32` for `"quickr"`, which has no single precision -- and they follow the
#' backend in force ([`default_backend()`]): `with_backend("quickr", ...)` commits
#' a double to `f64`. The options `anvl.default_float` and `anvl.default_int`
#' override them on every backend, e.g. `options(anvl.default_float = "f64")`;
#' `local_default_dtypes()` and `with_default_dtypes()` set them for a scope:
#' `local_default_dtypes(c(float = "f64"))`.
#'
#' The defaults decide only what a value becomes when *nothing else does*: an R
#' value that meets a typed array of its own category still takes that array's
#' data type, whatever the default (`vignette("type-promotion")`). A float
#' default is `"f32"` or `"f64"`, an integer default `"i32"` or `"i64"`. A
#' compiled program is keyed on the defaults it was compiled under, so changing
#' them never serves a stale program.
#'
#' Inside a [`jit()`]ted body the keyed defaults are the *baseline*, and a
#' scoped override applies to its scope -- so one program can use different
#' precisions in different parts of itself, including inside a helper the scope
#' calls. That is sound because an override written in the body belongs to the
#' program: it traces the same way every time, whatever the baseline. Switching
#' the *backend* inside a traced body changes nothing, since a program is
#' compiled for one backend.
#'
#' A scope covers the values *built* inside it. A bare R value handed back out
#' of one has not committed to anything yet, and takes the default in force
#' wherever it is eventually used -- the same rule that lets it take the data
#' type of whatever array it meets (see `vignette("type-promotion")`).
#'
#' @param dtypes (named `character()` | named `list()`)\cr
#'   The defaults to set, by category: element `float` (`"f32"` or `"f64"`)
#'   and/or element `int` (`"i32"` or `"i64"`), each a string or a
#'   [`DataType`]. A category that is not named is left as it is.
#' @param envir (`environment`)\cr
#'   The environment to scope the change to.
#' @param code An expression to evaluate with the given defaults.
#' @return `default_dtypes()` returns a named `list` with elements `float` and
#'   `int`, each a [`DataType`]: the defaults in force where it is called, which
#'   inside a [`jit()`]ted body is the trace's baseline and any override over
#'   it.
#'   `local_default_dtypes()` returns the previous values of the options it set,
#'   invisibly. `with_default_dtypes()` returns the result of evaluating `code`.
#' @seealso [`default_backend()`], [`peek_dtype()`]
#' @examplesIf pjrt::plugins_downloaded()
#' default_dtypes()
#' dtype(nv_array(1.5))
#' with_default_dtypes(c(float = "f64"), dtype(nv_array(1.5)))
#' # A value that meets a typed array still takes that array's data type
#' with_default_dtypes(c(float = "f64"), dtype(nv_array(1, dtype = "f32") + 1.5))
#' # Different precisions in different parts of one compiled program
#' f <- jit(function(x) {
#'   list(single = x * 1.5, double = with_default_dtypes(c(float = "f64"), x * 1.5))
#' })
#' f(nv_array(1L, dtype = "i32"))
#' @export
default_dtypes <- function() {
  current_default_dtypes()
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
  resolver <- function(category) {
    option <- default_dtype_option(category)
    fallback <- as.character(registered[[category]])
    allowed <- default_dtype_choices[[category]]
    what <- sprintf("Option {.code %s}", option)
    function() {
      value <- getOption(option)
      if (is.null(value)) {
        fallback
      } else if (is.character(value) && length(value) == 1L && value %in% allowed) {
        # The common shape of a set option; the full check is for anything else.
        as.character(value)
      } else {
        as.character(check_default_dtype(value, category, what))
      }
    }
  }
  float <- resolver("float")
  int <- resolver("int")
  function() c(float = float(), int = int())
}

# The inverse, for the compile callback: the defaults a trace is pinned to,
# from the context the dispatcher keyed the entry on. `NULL` stays `NULL`.
default_dtypes_from_key <- function(key) {
  if (is.null(key)) {
    return(NULL)
  }
  list(float = as_dtype(key[["float"]]), int = as_dtype(key[["int"]]))
}

# The options `dtypes` -- a named list or character vector with elements
# `float` and/or `int` -- asks to set, validated.
default_dtype_options <- function(dtypes) {
  categories <- names(dtypes)
  if (
    !(is.list(dtypes) || is.character(dtypes)) ||
      (length(dtypes) && (is.null(categories) || !all(categories %in% names(default_dtype_choices))))
  ) {
    cli_abort(c(
      "{.arg dtypes} must be a named list or character vector with elements {.val float} and/or {.val int}.",
      x = "Got {.obj_type_friendly {dtypes}} with names {.val {categories}}."
    ))
  }
  if (anyDuplicated(categories)) {
    duplicated <- unique(categories[duplicated(categories)])
    cli_abort(c(
      "{.arg dtypes} names {cli::qty(duplicated)}{?a category/categories} more than once: {.val {duplicated}}.",
      i = "Give each of {.val float} and {.val int} at most one data type."
    ))
  }
  opts <- lapply(categories, function(category) {
    as.character(check_default_dtype(dtypes[[category]], category, sprintf("{.code dtypes$%s}", category)))
  })
  names(opts) <- vapply(categories, default_dtype_option, character(1L))
  opts
}

#' @rdname default_dtypes
#' @export
local_default_dtypes <- function(dtypes, envir = parent.frame()) {
  withr::local_options(default_dtype_options(dtypes), .local_envir = envir)
}

#' @rdname default_dtypes
#' @export
with_default_dtypes <- function(dtypes, code) {
  withr::with_options(default_dtype_options(dtypes), code)
}

# The default dtypes (see `default_dtypes()`) in force here.
#
# Eagerly that is the override option over the registered default of the
# backend in force -- the pair the next dispatch keys its program on, since
# every operation runs on that backend.
#
# Inside a trace the *baseline* is the pair the dispatcher did key the program
# on, so switching the backend in a traced body changes nothing: the program is
# compiled for one backend. An override is still honoured, and applies to its
# scope: one written in the body is part of the program, so it traces the same
# way every time that key does, and a program may use different precisions in
# different parts of itself.
current_default_dtypes <- function() {
  desc <- globals[["CURRENT_DESCRIPTOR"]]
  if (is.null(desc)) {
    return(effective_default_dtypes(default_backend()))
  }
  pinned <- desc$default_dtypes
  list(
    float = option_default_dtype("float") %||% pinned$float,
    int = option_default_dtype("int") %||% pinned$int
  )
}

default_dtype <- function(x, defaults = current_default_dtypes()) {
  if (!is.numeric(x) && !is.logical(x)) {
    cli_abort("No default type for {.obj_type_friendly {x}}.")
  }
  default_dtype_r(typeof(x), defaults)
}

# `dtype`, or -- when it is `NULL` and `data` is an R value that has a default --
# the default `data` commits to. Anything else (a `PJRTBuffer`, a raw vector)
# is left for the backend to handle.
resolve_default_dtype <- function(data, dtype, defaults = current_default_dtypes()) {
  if (is.null(dtype) && (is.numeric(data) || is.logical(data))) {
    return(default_dtype(data, defaults))
  }
  dtype
}

# The dtype an R value of this storage type commits to when nothing in the
# program tells it what it is. The single place that decision is made;
# `defaults` is the pair (see `default_dtypes()`) the current computation is
# pinned to: the trace's while tracing, the default backend's otherwise.
default_dtype_r <- function(r_type, defaults = current_default_dtypes()) {
  switch(
    r_type,
    double = defaults$float,
    integer = defaults$int,
    logical = as_dtype("bool"),
    cli_abort("No default type for R type {.val {r_type}}")
  )
}
