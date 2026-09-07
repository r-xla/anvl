# The default data types: what an R double and an R integer commit to when
# nothing else decides one. Registered per backend, overridden by one global
# option, and pinned on a trace as the pair the dispatcher keyed the compiled
# program on.

# The dtypes a default may be set to, per category. `f16` / `bf16` are left out
# because the boundary cannot carry them, not because an R double could not be
# built at one: pjrt's dispatcher neither keys nor wraps them, and a program
# that reaches one fails at a different layer depending on how it got there.
# Lifting the restriction means fixing that boundary first. An integer default
# is limited to the signed dtypes an R integer builds at directly (see
# `rdata_builds_directly()`): a narrower one would make the upload of an R
# argument go through R's coercion, which wraps where the program's `convert`
# clamps, and an unsigned one cannot hold a negative R integer at all.
default_dtype_choices <- list(
  float = c("f32", "f64"),
  int = c("i32", "i64")
)

# The override lives in one option rather than one per category, so that it has
# the same shape as `default_dtypes()` reports and as the setters take.
default_dtypes_option <- "anvl.default_dtypes"

# What the option's value is called in an error, as cli markup.
option_what <- "The {.code anvl.default_dtypes} option"

# Validate one category's default. `source` says where it came from.
check_default_dtype <- function(x, category, source) {
  allowed <- default_dtype_choices[[category]]
  dtype <- tryCatch(as_dtype(x), error = function(e) NULL)
  if (is.null(dtype) || !(as.character(dtype) %in% allowed)) {
    cli_abort(
      c(
        "The {.field {category}} default must be one of {.val {allowed}}.",
        x = "Got {.val {x}}.",
        i = paste0("Set in ", source, ".")
      ),
      call = NULL
    )
  }
  dtype
}

# Validate a set of defaults -- the option's value, or a setter's argument --
# as a named character vector of canonical dtype names over a subset of the
# categories. `what` leads the error, `source` says where the value came from.
check_default_dtypes <- function(dtypes, what, source) {
  categories <- names(dtypes)
  if (
    !(is.list(dtypes) || is.character(dtypes)) ||
      (length(dtypes) && (is.null(categories) || !all(categories %in% names(default_dtype_choices))))
  ) {
    cli_abort(
      c(
        paste0(what, " must be a named list or character vector with elements {.val float} and/or {.val int}."),
        x = "Got {.obj_type_friendly {dtypes}} with names {.val {categories}}."
      ),
      call = NULL
    )
  }
  if (anyDuplicated(categories)) {
    duplicated <- unique(categories[duplicated(categories)])
    cli_abort(
      c(
        paste0(what, " names {cli::qty(duplicated)}{?a category/categories} more than once: {.val {duplicated}}."),
        i = "Give each of {.val float} and {.val int} at most one data type."
      ),
      call = NULL
    )
  }
  vapply(
    categories,
    function(category) as.character(check_default_dtype(dtypes[[category]], category, source)),
    character(1L)
  )
}

# The override in force, validated. Empty when the option is unset.
option_default_dtypes <- function() {
  value <- getOption(default_dtypes_option)
  if (is.null(value)) {
    return(character())
  }
  check_default_dtypes(value, option_what, paste0("option ", "{.code anvl.default_dtypes}"))
}

# `baseline` -- a list of DataTypes -- with the override applied over it. A
# category the override does not name keeps the baseline's data type, which is
# what lets a scope raise the float default without disturbing the integer one.
apply_default_dtypes <- function(baseline, override = option_default_dtypes()) {
  for (category in names(override)) {
    baseline[[category]] <- as_dtype(override[[category]])
  }
  baseline
}

# The effective pair for `backend`: the override over its registered defaults.
effective_default_dtypes <- function(backend) {
  apply_default_dtypes(registered_default_dtypes(backend))
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
#' `default_dtypes()` reports both categories at once; `default_float()` and
#' `default_int()` report one each, for where naming a single category reads
#' better than subsetting the pair.
#'
#' Each backend registers its own defaults -- `f32` / `i32` for `"pjrt"`, `f64`
#' / `i32` for `"quickr"`, which has no single precision -- and they follow the
#' backend in force ([`default_backend()`]): `with_backend("quickr", ...)` commits
#' a double to `f64`. The option `anvl.default_dtypes` overrides them on every
#' backend, e.g. `options(anvl.default_dtypes = c(float = "f64"))`;
#' `local_default_dtypes()` and `with_default_dtypes()` set it for a scope, and
#' name only the categories they change: `local_default_dtypes(c(float =
#' "f64"))` leaves the integer default alone.
#'
#' The defaults decide only what a value becomes when *nothing else does*: an R
#' value that meets a typed array of its own category still takes that array's
#' data type, whatever the default (`vignette("type-promotion")`). A float
#' default is `"f32"` or `"f64"`, an integer default `"i32"` or `"i64"`; a
#' backend that cannot represent the one you set says so rather than quietly
#' falling back, so both `"f32"` and `"i64"` are errors on `"quickr"`, which
#' has neither. A compiled program is keyed on the defaults it was compiled
#' under, so changing them never serves a stale program.
#'
#' Inside a [`jit()`]ted body the keyed defaults are the *baseline*, and a
#' scoped override applies to its scope. What it changes is what an
#' **uncommitted** R value in that scope commits to: a literal, an R array, or
#' a constructor called without a `dtype`. It does **not** change the data type
#' of an operand that already has one, so it cannot raise the precision of
#' arithmetic on typed arrays -- `with_default_dtypes(c(float = "f64"), x * 2)`
#' is `f32` for an `f32` `x`. Convert those explicitly with [`nv_convert()`].
#' Switching the *backend* inside a traced body changes nothing either, since a
#' program is compiled for one backend.
#'
#' Only the baseline is part of the compilation cache key, so inside a traced
#' body the override has to be written out literally rather than read from a
#' variable. And because an R *argument* of a jitted function is uploaded at a
#' single data type for the whole program, an override that reaches such an
#' argument in one part of a body decides how the caller's value arrives for
#' every part of it.
#'
#' A scope covers the values *built* inside it. A bare R value handed back out
#' of one has not committed to anything yet, and takes the default in force
#' wherever it is eventually used -- the same rule that lets it take the data
#' type of whatever array it meets (see `vignette("type-promotion")`).
#'
#' Only the baseline is part of the compilation cache key, so an override
#' inside a body carries the same constraint as any other value the body reads
#' from its enclosing environment: it must not change between calls.
#' `with_default_dtypes(c(float = prec), ...)` with a `prec` that later changes
#' keeps serving the first program, exactly as a changing `dtype` argument
#' would.
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
#'   it. `default_float()` and `default_int()` return that one [`DataType`] of
#'   their category.
#'   `local_default_dtypes()` returns the previous values of the options it set,
#'   invisibly. `with_default_dtypes()` returns the result of evaluating `code`.
#' @seealso [`default_backend()`], [`peek_dtype()`]
#' @examplesIf pjrt::plugins_downloaded()
#' default_dtypes()
#' default_float()
#' default_int()
#' dtype(nv_array(1.5))
#' with_default_dtypes(c(float = "f64"), dtype(nv_array(1.5)))
#' # A value that meets a typed array still takes that array's data type
#' with_default_dtypes(c(float = "f64"), dtype(nv_array(1, dtype = "f32") + 1.5))
#' # Untyped values in one program can commit at different precisions
#' jit(function() {
#'   list(single = nv_fill(0, 2), double = with_default_dtypes(c(float = "f64"), nv_fill(0, 2)))
#' })()
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
    value <- getOption(default_dtypes_option)
    if (is.null(value)) {
      # The overwhelmingly common case: nothing set, so the backend decides.
      return(fallback)
    }
    override <- check_default_dtypes(value, option_what, paste0("option ", "{.code anvl.default_dtypes}"))
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

# The option value `dtypes` asks for, merged over whatever is already set so
# that a scope naming one category leaves the other where it was.
merged_default_dtypes <- function(dtypes) {
  current <- option_default_dtypes()
  new <- check_default_dtypes(dtypes, "{.arg dtypes}", "{.arg dtypes}")
  current[names(new)] <- new
  setNames(list(current), default_dtypes_option)
}

#' @rdname default_dtypes
#' @export
local_default_dtypes <- function(dtypes, envir = parent.frame()) {
  check_literal_override(substitute(dtypes))
  withr::local_options(merged_default_dtypes(dtypes), .local_envir = envir)
}

#' @rdname default_dtypes
#' @export
with_default_dtypes <- function(dtypes, code) {
  check_literal_override(substitute(dtypes))
  withr::with_options(merged_default_dtypes(dtypes), code)
}

# Inside a trace an override belongs to the program but not to its compilation
# cache key, so it has to be the same on every call: written out literally.
# Read from a variable it would let the program's data types depend on session
# state the key cannot see -- and, since the key does carry the inputs' shapes,
# on the shape a call happens to be made with, which is how the same source
# text could give two precisions.
check_literal_override <- function(expr) {
  if (!currently_tracing() || is_literal_dtypes(expr)) {
    return(invisible(NULL))
  }
  cli_abort(c(
    "Inside a {.fn jit}ted function the defaults must be written out literally.",
    x = "Got {.code {deparse1(expr)}}.",
    i = "Only the baseline is part of the compilation cache key, so an override read from a variable would let the program's data types depend on state the key cannot see.", # nolint
    i = "Write it out, as in {.code with_default_dtypes(c(float = \"f64\"), ...)}, or convert the values explicitly with {.fn nv_convert}." # nolint
  ))
}

# An expression whose value cannot differ between two calls: `c(float =
# "f64")`, `list(int = "i64")`, or a character vector written out.
is_literal_dtypes <- function(expr) {
  if (is.character(expr)) {
    return(TRUE)
  }
  if (!is.call(expr) || !is.symbol(expr[[1L]]) || !(as.character(expr[[1L]]) %in% c("c", "list"))) {
    return(FALSE)
  }
  args <- as.list(expr)[-1L]
  length(args) > 0L && all(vapply(args, is.character, logical(1L)))
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
  apply_default_dtypes(desc$default_dtypes)
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
