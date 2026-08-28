#' @title Type Promotion Rules
#' @description
#' Computes the common data type of two data types, following the rules
#' described in `vignette("type-promotion")`.
#'
#' R values entering a program have no data type of their own (see
#' [`RDataArray`]) and are not promoted with this: they *take* the dtype of
#' whatever they are combined with, and commit to a default only when nothing
#' claims them.
#'
#' @param lhs_dtype ([`tengen::DataType`])\cr
#'   The left-hand side type.
#' @param rhs_dtype ([`tengen::DataType`])\cr
#'   The right-hand side type.
#' @return ([`tengen::DataType`])
#' @examples
#' common_dtype("i32", "f32")
#' common_dtype("i32", "i64")
#' @export
common_dtype <- function(lhs_dtype, rhs_dtype) {
  # REVIEW: Maybe it would still be good to support common_dtype("double", "i32") or something?
  # (it should resolve to the default double dtype in this case I guess)
  # Actually, I am not so sure this is a good idea.
  # RESPONSE: Left as it is, agreeing with your second thought. `common_dtype()`
  # is the user-facing statement of the data-type lattice, and an R storage type
  # is not a point in it: `"double"` against `"i32"` is `f32`, but `"double"`
  # against `"f64"` is `f64` and against `"i8"` is still `f32` -- that is a
  # different function of two arguments, and it already exists as
  # `promote_dt_rdata()` / `common_dtype_of()`, which is what the rules call.
  # Overloading one name with both would make the documented table wrong for
  # half its inputs.
  promote_dt_known(as_dtype(lhs_dtype), as_dtype(rhs_dtype))
}

#' @title Promotion Rules
#' @name promote_rule
#' @description
#' Which data type [`as_anvl_arrays()`] brings its inputs to. A rule is passed
#' as that function's `.promote` argument; without one, each input keeps the data
#' type it has and a bare R value takes its default.
#'
#' * `promote_common()` -- the common data type of the inputs
#'   (see [`common_dtype()`]). What a binary operator does.
#' * `promote_like()` -- the data type one particular input already has. For a
#'   function whose result type *is* one argument's type, and whose other
#'   arguments come along: `nv_clamp(min_val, x, max_val)` is `x`'s type.
#' * `promote_dtype()` -- a data type the function names itself, rather than one
#'   read off an input. For a function with an explicit `dtype` argument.
#' * `promote_yield()` -- the data type the inputs that *have* one already share.
#'   Unlike the other three it never converts an input that has a data type; only
#'   the R values move, and only within their own category. This is what the
#'   primitives use, and what a function wants when its arguments must agree but
#'   it will not widen the array it was given.
#'
#' @details
#' Inputs are *realized* at the target rather than converted to it: an R value
#' has no data type to convert from, so it is built at the target directly and
#' arrives with every digit it had (which is what keeps `x_f64 / sqrt(2)`
#' exact). An input that already has a data type is converted.
#'
#' @section Narrowing, and `force`:
#' The two rules that *name* a target -- `promote_like()` and `promote_dtype()`
#' -- refuse an input the target cannot hold, rather than narrowing it silently.
#' An input that has a data type must be promotable to the target
#' ([`common_dtype()`] of the two must be the target), and an R value must
#' *yield* to it, which it only does within its own category: an R double is
#' never built at an integer data type, whatever its value, because the same
#' answer has to hold for a jit argument nobody has seen.
#'
#' ```r
#' nv_clamp(0, nv_array(1L), 1.5)   # error: 1.5 has no place in an i32
#' nv_pad(nv_array(1L), 0L, 1L, 1L) # fine: write the literal in x's category
#' ```
#'
#' `force = TRUE` says the narrowing is the point and converts anyway. It is for
#' a function whose contract *is* the conversion, not a way past an error
#' message: prefer [`nv_convert()`] at the call site where the caller should see
#' it happening.
#'
#' `promote_common()` needs no such flag -- its target is one every input reaches
#' by construction -- and `promote_yield()` converts nothing that has a data type
#' at all.
#'
#' @section Yielding, and categories:
#' `promote_yield()` reads its target off the inputs rather than computing one:
#'
#' * If exactly one data type is present among the inputs, that is the target and
#'   the R values take it.
#' * If none is -- every input is an R value -- they must all be of the same R
#'   storage type, and they take its default (`f32` / `i32` / `bool`). A mix of
#'   storage types has no data type to agree on and is an error.
#' * If more than one is present, the inputs cannot be brought together without
#'   converting one of them, which this rule does not do. Nothing is realized,
#'   and whatever consumes the values reports the mismatch.
#'
#' An R value only ever yields **within its own category**: a double becomes a
#' float, an integer an integer, a logical a `bool`. `promote_yield()` on an
#' `f64` input and a `1L` is an error, where a `1` gives `f64` -- crossing a
#' category is promotion, which is what the other rules are for.
#'
#' `only` restricts the rule to some of the inputs. The rest are still aligned
#' onto one device and converted, just not to the target -- for an argument that
#' has no business taking part in the promotion, such as [`nv_ifelse()`]'s
#' `pred`, which stays a `bool`.
#'
#' @section Several groups in one call:
#' `promote_grouped()` combines rules, which is how one call promotes several
#' groups of arguments independently -- `x` and `y` to their common data type,
#' `a` and `b` to theirs:
#'
#' ```r
#' as_anvl_arrays(
#'   x = x, y = y, a = a, b = b,
#'   .promote = promote_grouped(
#'     promote_common(only = c("x", "y")),
#'     promote_common(only = c("a", "b"))
#'   )
#' )
#' ```
#'
#' The rules may be of different kinds, and they are all resolved against the
#' call's arguments before any of them is applied, so one group's target never
#' depends on another's having been applied first. They must cover disjoint
#' arguments -- a value cannot be brought to two data types -- and an argument
#' no rule names is aligned and converted, as it would be without a rule at
#' all.
#'
#' @param arg (`character(1)` | `numeric(1)`)\cr
#'   Which input to take the data type from: its name in the
#'   [`as_anvl_arrays()`] call, or its position. Naming it needs the call's
#'   arguments to be named.
#' @param dtype ([`tengen::DataType`] | `character(1)`)\cr
#'   The data type to bring the inputs to.
#' @param only (`NULL` | `character()` | `numeric()`)\cr
#'   Which inputs the rule applies to, by name or position. `NULL` (default) is
#'   all of them.
#' @param force (`logical(1)`)\cr
#'   Convert an input the target data type cannot hold, instead of raising an
#'   error. `FALSE` (default) is what you want unless narrowing is the
#'   function's contract. See the *Narrowing* section.
#' @param ... ([`PromoteRule`][promote_rule])\cr
#'   For `promote_grouped()`: the rules to apply, each covering its own group of
#'   arguments.
#' @return (`PromoteRule`)
#' @seealso [as_anvl_arrays()], [nv_promote_to_common()], [common_dtype()]
#' @examplesIf pjrt::plugins_downloaded()
#' as_anvl_arrays(nv_array(1L), 1.5, .promote = promote_common())
#' # yielding: the R value takes the array's dtype, the array never moves
#' as_anvl_arrays(nv_array(1, dtype = "f64"), 1.5, .promote = promote_yield())
#' as_anvl_arrays(x = nv_array(1L), y = 1L, .promote = promote_like("x"))
#' as_anvl_arrays(nv_array(1L), 1.5, .promote = promote_dtype("f64"))
#' # a target that cannot hold an input is an error, unless that is the point
#' try(as_anvl_arrays(x = nv_array(1L), y = 1.5, .promote = promote_like("x")))
#' as_anvl_arrays(x = nv_array(1L), y = 1.5, .promote = promote_like("x", force = TRUE))
#' # `pred` is aligned and converted, but keeps out of the promotion
#' as_anvl_arrays(
#'   pred = nv_array(TRUE),
#'   a = nv_array(1L),
#'   b = 1.5,
#'   .promote = promote_common(only = c("a", "b"))
#' )
#' # two groups, promoted independently
#' as_anvl_arrays(
#'   x = nv_array(1L), y = 1.5,
#'   a = nv_array(1L, dtype = "i8"), b = 2L,
#'   .promote = promote_grouped(
#'     promote_common(only = c("x", "y")),
#'     promote_common(only = c("a", "b"))
#'   )
#' )
NULL

#' @rdname promote_rule
#' @export
promote_common <- function(only = NULL) {
  PromoteRule("common", only = only)
}

#' @rdname promote_rule
#' @export
promote_like <- function(arg, only = NULL, force = FALSE) {
  assert_arg_ref(arg, "arg", len = 1L)
  assert_flag(force)
  PromoteRule("like", only = only, arg = arg, force = force)
}

#' @rdname promote_rule
#' @export
promote_dtype <- function(dtype, only = NULL, force = FALSE) {
  assert_flag(force)
  PromoteRule("dtype", only = only, dtype = as_dtype(dtype), force = force)
}

# What the primitives whose arrayish arguments must agree declare, and the reason
# `prim_mul(x_f64, 2)` works: a primitive must not widen the array it was handed
# -- that is the `nv_*` layer's job -- but it does have to say what the R values
# among its operands become, and "the data type the real operands already have"
# is the only answer that does not invent one.
#' @rdname promote_rule
#' @export
promote_yield <- function(only = NULL) {
  PromoteRule("yield", only = only)
}

#' @rdname promote_rule
#' @export
promote_grouped <- function(...) {
  rules <- list(...)
  ok <- length(rules) > 0L &&
    all(vapply(rules, function(r) is_promote_rule(r) && r$kind != "grouped", logical(1L)))
  if (!ok) {
    cli_abort(c(
      "{.fn promote_grouped} takes promotion rules, one per group of arguments.",
      i = "Build them with {.fn promote_common}, {.fn promote_like}, {.fn promote_dtype} or {.fn promote_yield}; groups do not nest." # nolint
    ))
  }
  structure(list(kind = "grouped", only = NULL, rules = rules), class = "PromoteRule")
}

# A tagged union over the four rules. The tag rather than four S3 subclasses:
# the whole of the behaviour is `resolve_promote_rule()`, and one function that
# shows them side by side is easier to read than four one-line methods.
PromoteRule <- function(kind, only, ...) {
  if (!is.null(only)) {
    assert_arg_ref(only, "only")
  }
  structure(list(kind = kind, only = only, ...), class = "PromoteRule")
}

is_promote_rule <- function(x) {
  inherits(x, "PromoteRule")
}

# A reference to one of `as_anvl_arrays()`'s inputs: a name, or a position.
assert_arg_ref <- function(x, what, len = NULL) {
  ok <- (is.character(x) || (is.numeric(x) && all(x == trunc(x)))) &&
    !anyNA(x) &&
    (is.null(len) || length(x) == len)
  if (!ok) {
    cli_abort(
      "{.arg {what}} must be {if (is.null(len)) 'names or positions' else 'the name or position'} of {cli::qty(len %||% 2L)}{?an/} argument{?s}, not {.cls {class(x)}}." # nolint
    )
  }
  invisible(x)
}

#' @export
format.PromoteRule <- function(x, ...) {
  detail <- switch(
    x$kind,
    common = ,
    yield = "",
    like = sprintf("(%s%s)", format_arg_ref(x$arg), if (isTRUE(x$force)) ", force" else ""),
    dtype = sprintf("(%s%s)", repr(x$dtype), if (isTRUE(x$force)) ", force" else ""),
    grouped = sprintf("(%s)", paste(vapply(x$rules, format, character(1L)), collapse = ", "))
  )
  only <- if (is.null(x$only)) "" else sprintf(" only %s", format_arg_ref(x$only))
  sprintf("<promote_%s%s%s>", x$kind, detail, only)
}

format_arg_ref <- function(x) {
  if (is.character(x)) {
    return(paste0("\"", x, "\"", collapse = ", "))
  }
  paste(x, collapse = ", ")
}

#' @export
print.PromoteRule <- function(x, ...) {
  cat(format(x), "\n")
  invisible(x)
}

# Carry out resolved rules: each argument a rule covers is realized at that
# rule's target dtype. The one place a promotion rule is applied, shared by
# `as_anvl_arrays()` and the primitive layer -- `...` is how the former passes
# the device it aligned on.
#
# A rule that could not settle on a dtype covers no arguments, so it is a no-op
# here and whatever consumes the values reports the mismatch.
apply_promote_rules <- function(args, resolved, ...) {
  for (rule in resolved) {
    if (isTRUE(rule$strict)) {
      for (i in rule$positions) {
        assert_promotes_to(args[[i]], rule$dtype, args, i)
      }
    }
    args[rule$positions] <- lapply(args[rule$positions], realize_at, dtype = rule$dtype, ...)
  }
  args
}

# Whether `x` reaches `dtype` without losing what it holds: a value that has a
# data type must be promotable to it, and an R value must *yield* to it -- a
# double does not become an integer, whatever its value, because the check has
# to hold for a jit argument whose value nobody has seen.
assert_promotes_to <- function(x, dtype, args, i) {
  aval <- to_abstract(x)
  what <- if (nzchar(rlang::names2(args)[[i]])) {
    sprintf("`%s`", rlang::names2(args)[[i]])
  } else {
    sprintf("argument %d", i)
  }
  target <- as.character(dtype)
  if (is_rdata_array(aval)) {
    if (promote_dt_rdata(aval$default_dtype, dtype) == dtype) {
      return(invisible(NULL))
    }
    cli_abort(
      c(
        "Cannot bring {what} to data type {.val {target}}.",
        x = "It is an R {aval$r_type}, which is only ever built at a data type of its own category: a double becomes a float, an integer an integer, a logical a {.val bool}.", # nolint
        i = "Write it in the target's category (e.g. {.code 0L} for an integer data type), convert it with {.fn nv_convert}, or ask for the conversion with {.code force = TRUE}." # nolint
      ),
      call = NULL
    )
  }
  if (promotable_to(aval$dtype, dtype)) {
    return(invisible(NULL))
  }
  cli_abort(
    c(
      "Cannot bring {what} to data type {.val {target}}.",
      x = "{.val {as.character(aval$dtype)}} is not promotable to {.val {target}}.",
      i = "Convert it explicitly with {.fn nv_convert}, or ask for the conversion with {.code force = TRUE}." # nolint
    ),
    call = NULL
  )
}

# Resolve every rule of a call against its arguments, before any of them is
# applied: a later rule reads the dtypes the earlier ones have not changed yet.
# `.promote` is one rule, possibly a `promote_grouped()` of several -- which is
# how a call promotes groups independently, so those must cover disjoint
# arguments. Returns a list of `list(dtype, positions)`.
resolve_promote_rules <- function(promote, args) {
  if (!is_promote_rule(promote)) {
    cli_abort(c(
      "{.arg .promote} must be a promotion rule, not {.cls {class(promote)}}.",
      i = "Build one with {.fn promote_common}, {.fn promote_like}, {.fn promote_dtype} or {.fn promote_yield}, and combine several with {.fn promote_grouped}." # nolint
    ))
  }
  rules <- if (promote$kind == "grouped") promote$rules else list(promote)
  resolved <- lapply(rules, resolve_promote_rule, args = args)
  positions <- unlist(lapply(resolved, `[[`, "positions"))
  clashing <- unique(positions[duplicated(positions)])
  if (length(clashing)) {
    named <- rlang::names2(args)[clashing]
    cli_abort(c(
      "More than one promotion rule covers the same argument.",
      x = "{cli::qty(length(clashing))}Argument{?s} {.val {ifelse(nzchar(named), named, as.character(clashing))}} {?is/are} covered twice.", # nolint
      i = "An argument can only be brought to one data type, so the rules must name disjoint sets with {.arg only}." # nolint
    ))
  }
  resolved
}

# The dtype one rule brings its inputs to, and which of them it applies to.
# Returns `list(dtype, positions)`; `positions` indexes `args`.
resolve_promote_rule <- function(rule, args) {
  positions <- if (is.null(rule$only)) seq_along(args) else arg_positions(rule$only, args, "only")
  if (rule$kind == "yield") {
    return(resolve_yield(args, positions))
  }
  # An R value among the inputs yields: it contributes the dtype it would take
  # on its own and takes the others' where there is one.
  dtype <- switch(
    rule$kind,
    common = do.call(common_dtype_of, args[positions]),
    like = do.call(common_dtype_of, args[arg_positions(rule$arg, args, "arg")]),
    dtype = rule$dtype
  )
  # `promote_common()` computes a target every input reaches by construction;
  # the two rules that *name* one have to check that they do, unless the caller
  # said the conversion is what it wants (`force`).
  list(dtype = dtype, positions = positions, strict = !isTRUE(rule$force))
}

# `promote_yield()`: read the target off the inputs that already have a dtype,
# never move them, and let the R values take it -- within their own category.
resolve_yield <- function(args, positions) {
  avals <- lapply(args[positions], to_abstract)
  is_r <- vapply(avals, is_rdata_array, logical(1L))
  settled <- unique(lapply(avals[!is_r], peek_dtype))
  if (length(settled) > 1L) {
    # The inputs cannot be brought together without converting one of them.
    # Realize nothing and let whatever consumes them report the mismatch.
    return(list(dtype = NULL, positions = integer()))
  }
  if (!length(settled)) {
    # Every input is an R value. They agree only if they are of one storage
    # type; there is no data type for a mix of them to meet at.
    r_types <- unique(vapply(avals[is_r], function(a) a$r_type, character(1L)))
    if (length(r_types) > 1L) {
      cli_abort(
        c(
          "The R values here have no data type to agree on.",
          x = "Got {.val {r_types}} values, which belong to different data type categories.",
          i = "Give one of them a data type with {.fn nv_scalar} or {.fn nv_array}, or use an operation that promotes across categories." # nolint
        ),
        call = NULL
      )
    }
    return(list(dtype = default_dtype_r(r_types), positions = positions))
  }
  target <- settled[[1L]]
  # Each R value must be able to take the target without leaving its category.
  for (aval in avals[is_r]) {
    r_type <- aval$r_type
    if (!rdata_in_category(r_type, target)) {
      cli_abort(
        c(
          "An R {r_type} cannot be used at the {.val {as.character(target)}} data type here.",
          i = "A literal is only ever built at a data type of its own category: a double becomes a float, an integer an integer, a logical a {.val bool}.", # nolint
          i = "Use an operation that promotes across categories, or convert explicitly with {.fn nv_convert}."
        ),
        call = NULL
      )
    }
  }
  list(dtype = target, positions = positions)
}

# Resolve argument references -- names or positions -- against the call's
# arguments.
arg_positions <- function(ref, args, what) {
  if (!is.character(ref)) {
    assert_integerish(ref, lower = 1L, upper = length(args), .var.name = what)
    return(as.integer(ref))
  }
  found <- match(ref, rlang::names2(args))
  if (anyNA(found)) {
    cli_abort(c(
      "{.arg {what}} names {?an argument/arguments} this call does not have: {.val {ref[is.na(found)]}}.",
      i = "The arguments are {.val {rlang::names2(args)}}.",
      i = "Referring to one by name needs them named, e.g. {.code as_anvl_arrays(x = x, y = y, .promote = promote_like(\"x\"))}." # nolint
    ))
  }
  found
}

# The common dtype of several arrayish values, the one every operand of an
# operation is brought to. An R value yields: it takes the dtype of the values
# it meets, and contributes only the dtype it would commit to when it meets
# nothing but other R values.
# For internal use.
common_dtype_of <- function(...) {
  args <- list(...)
  if (length(args) == 0L) {
    cli_abort("No arguments provided")
  }
  cdt <- NULL
  cdt_is_rdata <- TRUE
  for (arg in args) {
    aval <- to_abstract(arg)
    is_rdata <- is_rdata_array(aval)
    dt <- peek_dtype(aval)
    if (is.null(cdt)) {
      cdt <- dt
      cdt_is_rdata <- is_rdata
      next
    }
    if (cdt_is_rdata && is_rdata) {
      cdt <- promote_dt_known(cdt, dt)
    } else if (cdt_is_rdata) {
      cdt <- promote_dt_rdata(cdt, dt)
      cdt_is_rdata <- FALSE
    } else if (is_rdata) {
      cdt <- promote_dt_rdata(dt, cdt)
    } else {
      cdt <- promote_dt_known(cdt, dt)
    }
  }
  cdt
}


# The ordering the whole RData design turns on: `bool` < integer < float. An R
# value may only be *built* within its own category (`rdata_in_category()` asks
# this for equality), and it only *yields* to a dtype at least as high
# (`promote_dt_rdata()` asks it for the maximum).
dtype_category <- function(dtype) {
  if (is_dtype_bool(dtype)) {
    1L
  } else if (is_dtype_float(dtype)) {
    3L
  } else {
    2L
  }
}

# What an R value becomes when it meets a value that has a dtype. It yields --
# an R double meeting an `f16` array becomes `f16`, not the default float --
# except where the dtype's category cannot hold it: a value from a float R type
# meeting an integer dtype stays a float, and anything meeting `bool` stays
# itself. `rdtype` is the dtype the R value would commit to on its own.
promote_dt_rdata <- function(rdtype, dtype) {
  if (dtype_category(dtype) >= dtype_category(rdtype)) dtype else rdtype
}

promote_dt_known <- function(dt1, dt2) {
  if (dt1 == dt2) {
    return(dt1)
  }
  if (is_dtype_bool(dt1)) {
    return(dt2)
  }
  if (is_dtype_bool(dt2)) {
    return(dt1)
  }
  if (is_dtype_float(dt1)) {
    if (is_dtype_float(dt2)) {
      return(as_dtype(paste0("f", max(dtype_width(dt1), dtype_width(dt2)))))
    }
    # bools and integers are cast to the float
    return(dt1)
  }
  if (is_dtype_float(dt2)) {
    return(dt2)
  }
  if (is_dtype_int(dt1)) {
    if (is_dtype_int(dt2)) {
      return(as_dtype(paste0("i", max(dtype_width(dt1), dtype_width(dt2)))))
    }
    if (dtype_width(dt2) < dtype_width(dt1)) {
      # the int can hold the unsigned int
      return(dt1)
    }
    # int can't hold the unsigned int
    # we use signed int, but increase bits of unsigned int
    # this can lead to overflows then we have uint64 but this can't be avoided
    return(as_dtype(paste0("i", min(64L, dtype_width(dt2) * 2L))))
  }
  if (is_dtype_int(dt2)) {
    if (is_dtype_uint(dt1)) {
      if (dtype_width(dt2) > dtype_width(dt1)) {
        return(dt2)
      }
      return(as_dtype(paste0("i", min(64L, dtype_width(dt1) * 2L))))
    }
    cli_abort("internal error")
  }
  # both are unsigned
  as_dtype(paste0("ui", max(dtype_width(dt1), dtype_width(dt2))))
}

default_dtype <- function(x) {
  if (!is.numeric(x) && !is.logical(x)) {
    cli_abort("No default type for: {.class class(x)[1L]}")
  }
  default_dtype_r(typeof(x))
}

# The dtype an R value of this storage type commits to when nothing in the
# program tells it what it is. The single place that decision is made.
default_dtype_r <- function(r_type) {
  switch(
    r_type,
    double = as_dtype("f32"),
    integer = as_dtype("i32"),
    logical = as_dtype("bool"),
    cli_abort("No default type for R type {.val {r_type}}")
  )
}

promotable_to <- function(from, to) {
  if (identical(from, to)) {
    return(TRUE)
  }
  common_dtype(from, to) == to
}
