#' @title Type Promotion Rules
#' @description
#' Computes the common data type of two data types, following the rules
#' described in `vignette("type-promotion")`.
#'
#' R values entering a program have no data type of their own (see
#' [`RData`]) and are not promoted with this: they *take* the dtype of
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
  promote_dt_known(as_dtype(lhs_dtype), as_dtype(rhs_dtype))
}

#' @title Promotion Rules
#' @name promote_rule
#' @description
#' Which data type [`as_anvl_arrays()`] brings its inputs to. A rule is passed
#' as that function's `.promote` argument; without one, each input keeps the data
#' type it has and a bare R value takes its default.
#'
#' A rule is a **function of the call's arguments** that answers with the data
#' type each one is brought to (see the *Writing a rule* section).
#'
#' `promote_common()`, `promote_like()`, `promote_dtype()` and `promote_yield()`
#' are the four anvl builds for itself, and `promote_grouped()` combines them;
#' each says below what it decides. A package with a promotion of its own writes
#' another and passes it the same way.
#'
#' @details
#' Inputs are *realized* at the target rather than converted to it: an R value
#' has no data type to convert from, so it is built at the target directly and
#' arrives with every digit it had (which is what keeps `x_f64 / sqrt(2)`
#' exact). An input that already has a data type is converted.
#'
#' @section Writing a rule:
#' A rule is a function of one argument -- the call's arguments, as a list, named
#' where the call named them and holding each value exactly as the caller passed
#' it. It returns a list of the same length: the data type each argument is
#' realized at, or `NULL` for one it leaves where it is.
#'
#' ```r
#' # every argument at the widest float in the call, and never below f32
#' promote_widest_float <- function(args) {
#'   widths <- vapply(args, function(a) {
#'     dt <- peek_dtype(to_abstract(a))
#'     if (is_dtype_float(dt)) dtype_width(dt) else 0L
#'   }, integer(1))
#'   rep(list(as_dtype(paste0("f", max(c(32L, widths))))), length(args))
#' }
#' as_anvl_arrays(nv_array(1L), 2.5, .promote = promote_widest_float)
#' ```
#'
#' An argument may still be a bare R value when the rule sees it, which is the
#' point -- that is what lets the rule decide what the value becomes. So ask it
#' for a data type with [`peek_dtype()`] rather than `dtype()`, which an R value
#' has nothing to answer with.
#'
#' A rule is asked once per call, before anything is realized, so it always sees
#' the arguments as they were passed. [`promote_grouped()`] combines rules of any
#' kind, anvl's own and yours together.
#'
#' The framework realizes what the rule names and checks nothing about it: a rule
#' that names a data type an argument cannot hold narrows it silently. The two
#' built-in rules that name a target refuse that themselves, which is what the
#' next section describes.
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
#' @section Falling back:
#' `promote_common()` reads its target off the inputs, and an input that is a
#' bare R value has none to give -- so a call whose arguments are *all* R values
#' has nothing to read, and they commit to the default of their R storage type
#' (`f32` / `i32` / `bool`). `fallback` is the data type to settle on there
#' instead, for a function that has a type in mind but would rather take one
#' from its arguments:
#'
#' ```r
#' # nothing brings a data type: the fallback decides
#' as_anvl_arrays(1, 2L, .promote = promote_common(fallback = "f64"))
#' # an argument does: the fallback is ignored, whatever it named
#' as_anvl_arrays(nv_array(1), 2L, .promote = promote_common(fallback = "f64"))
#' ```
#'
#' It is a fallback and not a floor: it applies only where nothing else claims
#' the R values, so an input that *has* a data type wins over it even when that
#' data type is narrower or of another category. A function that needs the
#' result to be of some category regardless -- a sampler that must draw at a
#' float -- checks that for itself.
#'
#' The R values yield to a fallback the way they yield to any data type, within
#' their own category (see the previous section): a fallback of `"i32"` on a
#' call holding an R double leaves it a float.
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
#' The rules may be of different kinds -- anvl's own and yours together, and a
#' group is itself a rule, so groups nest. They are all asked before any of their
#' answers is used, so one group's target never depends on another's having been
#' applied first. They must cover disjoint arguments -- a value cannot be brought
#' to two data types -- and an argument no rule names is aligned and converted,
#' as it would be without a rule at all.
#'
#' @param only (`NULL` | `character()` | `numeric()`)\cr
#'   Which inputs the rule applies to, by name or position. `NULL` (default) is
#'   all of them.
#' @param force (`logical(1)`)\cr
#'   Convert an input the target data type cannot hold, instead of raising an
#'   error. `FALSE` (default) is what you want unless narrowing is the
#'   function's contract. See the *Narrowing* section.
#' @return (`function`)\cr
#'   The rule: a function of the call's arguments returning the data type each is
#'   brought to. The built-in ones carry a `PromoteRule` class so they say which
#'   rule they are when printed.
#' @seealso [as_anvl_arrays()], [nv_promote_to_common()], [common_dtype()]
#' @examplesIf pjrt::plugins_downloaded()
#' as_anvl_arrays(nv_array(1L), 1.5, .promote = promote_common())
#' # yielding: the R value takes the array's dtype, the array never moves
#' as_anvl_arrays(nv_array(1, dtype = "f64"), 1.5, .promote = promote_yield())
#' as_anvl_arrays(x = nv_array(1L), y = 1L, .promote = promote_like("x"))
#' as_anvl_arrays(nv_array(1L), 1.5, .promote = promote_dtype("f64"))
#' # with nothing to read a data type off, `fallback` decides
#' as_anvl_arrays(1, 2L, .promote = promote_common(fallback = "f64"))
#' as_anvl_arrays(nv_array(1L), 2L, .promote = promote_common(fallback = "f64"))
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

#' @description
#' `promote_common()` brings every input to their common data type
#' ([`common_dtype()`]) -- the one they all reach, chosen so nothing is lost on
#' the way. No input is privileged, which is what a binary operator wants:
#' `nv_add(x_i32, y_f64)` is `f64` whichever operand came first. An R value
#' among them yields rather than contributing a data type of its own, so an
#' `i8` array meeting `1L` stays `i8`.
#'
#' Where *no* input brings a data type, there is nothing to read one off and
#' they commit to the default of their R storage type; `fallback` names what
#' they take instead. See the *Falling back* section.
#' @param fallback (`NULL` | [`tengen::DataType`] | `character(1)`)\cr
#'   The data type to settle on when *every* input is a bare R value, in place
#'   of the default those would commit to on their own. `NULL` (default) leaves
#'   them their default.
#' @rdname promote_rule
#' @export
promote_common <- function(only = NULL, fallback = NULL) {
  assert_only(only)
  fallback <- if (!is.null(fallback)) as_dtype(fallback)
  new_promote_rule(
    function(args) {
      positions <- rule_positions(only, args, "only")
      if (!length(positions)) {
        return(dtypes_none(args))
      }
      dtype <- do.call(common_dtype_of, c(args[positions], list(.fallback = fallback)))
      dtypes_at(args, positions, dtype)
    },
    "common",
    only = only,
    fallback = fallback
  )
}

#' @description
#' `promote_like()` brings every input to the data type one particular input
#' already has. For a function whose result *is* one argument's type and whose
#' other arguments come along for the ride: `nv_clamp(min_val, x, max_val)` is
#' `x`'s type, and the bounds are built at it.
#'
#' Because it names a target rather than computing one, it refuses an input that
#' target cannot hold instead of narrowing it silently -- see the *Narrowing*
#' section, and `force` for the case where the narrowing is the point.
#' @param arg (`character(1)` | `numeric(1)`)\cr
#'   Which input to take the data type from: its name in the
#'   [`as_anvl_arrays()`] call, or its position. Naming it needs the call's
#'   arguments to be named.
#' @rdname promote_rule
#' @export
promote_like <- function(arg, only = NULL, force = FALSE) {
  assert_arg_ref(arg, "arg", len = 1L)
  assert_only(only)
  assert_flag(force)
  new_promote_rule(
    function(args) {
      dtype <- do.call(common_dtype_of, args[rule_positions(arg, args, "arg")])
      dtypes_named(args, rule_positions(only, args, "only"), dtype, force)
    },
    "like",
    only = only,
    arg = arg,
    force = force
  )
}

#' @description
#' `promote_dtype()` brings every input to a data type the function names
#' itself, rather than one read off an input. For a function with an explicit
#' `dtype` argument, where the caller's word settles the matter and the
#' arguments follow.
#'
#' Like [`promote_like()`] it names a target, so it refuses an input that target
#' cannot hold unless `force` says the narrowing is the contract.
#' @param dtype ([`tengen::DataType`] | `character(1)`)\cr
#'   The data type to bring the inputs to.
#' @rdname promote_rule
#' @export
promote_dtype <- function(dtype, only = NULL, force = FALSE) {
  assert_only(only)
  assert_flag(force)
  dtype <- as_dtype(dtype)
  new_promote_rule(
    function(args) {
      dtypes_named(args, rule_positions(only, args, "only"), dtype, force)
    },
    "dtype",
    only = only,
    dtype = dtype,
    force = force
  )
}

#' @description
#' `promote_yield()` reads the data type off the inputs that already have one
#' and never moves them: only the R values take it, and only within their own
#' category. It is the odd one out -- the other three may convert an input, and
#' this one never does.
#'
#' It is what the primitives declare, because a primitive must not widen the
#' array it was handed (that is the `nv_*` layer's job) while still having to say
#' what the R values among its operands become. It is also what an `nv_*`
#' function wants when its arguments must *agree* rather than meet: a padding
#' value that disagrees with the array is a mistake to report, not one to convert
#' away. If the inputs carry several data types it places nothing and leaves the
#' mismatch to whatever consumes them. See the *Yielding, and categories*
#' section.
#' @rdname promote_rule
#' @export
promote_yield <- function(only = NULL) {
  assert_only(only)
  new_promote_rule(
    function(args) resolve_yield(args, rule_positions(only, args, "only")),
    "yield",
    only = only
  )
}

#' @description
#' `promote_grouped()` applies several rules in one call, each to its own group
#' of arguments -- `x` and `y` to their common data type, `a` and `b` to theirs.
#' The rules may be of any kind, anvl's own and yours together, and a group is
#' itself a rule, so groups nest. They must cover disjoint arguments. See the
#' *Several groups in one call* section.
#' @param ... (`function`)\cr
#'   The rules to apply, each covering its own group of arguments.
#' @rdname promote_rule
#' @export
promote_grouped <- function(...) {
  rules <- list(...)
  if (!length(rules) || !all(vapply(rules, is.function, logical(1L)))) {
    cli_abort(c(
      "{.fn promote_grouped} takes promotion rules, one per group of arguments.",
      i = "A rule is a function of the call's arguments; build one with {.fn promote_common}, {.fn promote_like}, {.fn promote_dtype} or {.fn promote_yield}." # nolint
    ))
  }
  new_promote_rule(
    function(args) {
      # Every rule is asked before any of their answers is used, so one group's
      # target never depends on another's having been applied first.
      dtypes_merged(lapply(rules, resolve_promote, args = args), args)
    },
    "grouped",
    rules = rules
  )
}

# A rule is a function; the class is only so the built-in ones can say what they
# are when printed. Anything callable that answers the same way is a rule too --
# which is what lets a package promote by a rule anvl has never heard of.
new_promote_rule <- function(fn, kind, ...) {
  structure(fn, class = c("PromoteRule", "function"), kind = kind, spec = list(...))
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

assert_only <- function(only) {
  if (!is.null(only)) {
    assert_arg_ref(only, "only")
  }
  invisible(only)
}

#' @export
format.PromoteRule <- function(x, ...) {
  kind <- attr(x, "kind")
  spec <- attr(x, "spec")
  detail <- switch(
    kind,
    common = if (is.null(spec$fallback)) "" else sprintf("(fallback %s)", repr(spec$fallback)),
    yield = "",
    like = sprintf("(%s%s)", format_arg_ref(spec$arg), if (isTRUE(spec$force)) ", force" else ""),
    dtype = sprintf("(%s%s)", repr(spec$dtype), if (isTRUE(spec$force)) ", force" else ""),
    grouped = sprintf("(%s)", paste(vapply(spec$rules, format_rule, character(1L)), collapse = ", "))
  )
  only <- if (is.null(spec$only)) "" else sprintf(" only %s", format_arg_ref(spec$only))
  sprintf("<promote_%s%s%s>", kind, detail, only)
}

# How a rule is named in a message. A rule of anvl's own says which it is; one
# from elsewhere is just a function, and there is nothing to say about it.
format_rule <- function(x) {
  if (inherits(x, "PromoteRule")) format(x) else "<promotion rule>"
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

# Ask a rule what data type each argument is brought to, and check that it
# answered in the shape the contract asks for -- a bad answer is easier to
# report here, against the rule, than to trip over while realizing values.
resolve_promote <- function(promote, args) {
  if (!is.function(promote)) {
    cli_abort(c(
      "{.arg .promote} must be a promotion rule, not {.cls {class(promote)}}.",
      i = "A rule is a function of the call's arguments returning the data type each one is brought to.", # nolint
      i = "Build one with {.fn promote_common}, {.fn promote_like}, {.fn promote_dtype} or {.fn promote_yield}, and combine several with {.fn promote_grouped}." # nolint
    ))
  }
  dtypes <- promote(args)
  assert_rule_answer(dtypes, args, promote)
  dtypes
}

# The rule contract, checked: one entry per argument, each a data type or `NULL`.
assert_rule_answer <- function(dtypes, args, promote) {
  if (
    is.list(dtypes) &&
      length(dtypes) == length(args) &&
      all(vapply(dtypes, function(d) is.null(d) || is_dtype(d), logical(1L)))
  ) {
    return(invisible(dtypes))
  }
  cli_abort(c(
    "A promotion rule must answer with one data type per argument.",
    x = "{format_rule(promote)} returned {.obj_type_friendly {dtypes}} for {length(args)} argument{?s}.", # nolint
    i = "Return a list as long as the call's arguments, holding the data type each is brought to and {.code NULL} for one the rule leaves where it is." # nolint
  ))
}

# Realize every argument a rule places at the data type it named, and leave the
# rest as they are. What the primitive layer applies; `as_anvl_arrays()` does the
# same but converts the untouched ones as well.
promote_args <- function(args, promote, ...) {
  dtypes <- resolve_promote(promote, args)
  for (i in seq_along(args)) {
    if (!is.null(dtypes[[i]])) {
      args[[i]] <- realize_at(args[[i]], dtype = dtypes[[i]], ...)
    }
  }
  args
}

# The answer a rule gives when it places nothing.
dtypes_none <- function(args) {
  vector("list", length(args))
}

dtypes_at <- function(args, positions, dtype) {
  out <- dtypes_none(args)
  out[positions] <- list(dtype)
  out
}

# A rule that *names* a target, rather than reading one off the arguments, has
# to check that each argument reaches it -- unless the caller said the narrowing
# is the point.
dtypes_named <- function(args, positions, dtype, force) {
  if (!isTRUE(force)) {
    for (i in positions) {
      assert_promotes_to(args[[i]], dtype, args, i)
    }
  }
  dtypes_at(args, positions, dtype)
}

# Several rules' answers, laid over one another. They must place disjoint
# arguments -- a value cannot be brought to two data types.
dtypes_merged <- function(answers, args) {
  out <- dtypes_none(args)
  placed <- integer()
  for (dtypes in answers) {
    at <- which(!vapply(dtypes, is.null, logical(1L)))
    clashing <- intersect(placed, at)
    if (length(clashing)) {
      named <- rlang::names2(args)[clashing]
      cli_abort(c(
        "More than one promotion rule covers the same argument.",
        x = "{cli::qty(length(clashing))}Argument{?s} {.val {ifelse(nzchar(named), named, as.character(clashing))}} {?is/are} covered twice.", # nolint
        i = "An argument can only be brought to one data type, so the rules must name disjoint sets with {.arg only}." # nolint
      ))
    }
    placed <- c(placed, at)
    out[at] <- dtypes[at]
  }
  out
}

# How an argument is named in an error: by its own name where the call gave it
# one, by its position otherwise.
arg_label <- function(args, i) {
  nm <- rlang::names2(args)[[i]]
  if (nzchar(nm)) sprintf("`%s`", nm) else sprintf("argument %d", i)
}

# Whether `x` reaches `dtype` without losing what it holds: a value that has a
# data type must be promotable to it, and an R value must *yield* to it -- a
# double does not become an integer, whatever its value, because the check has
# to hold for a jit argument whose value nobody has seen.
assert_promotes_to <- function(x, dtype, args, i) {
  aval <- to_abstract(x)
  what <- arg_label(args, i)
  target <- as.character(dtype)
  if (is_rdata(aval)) {
    if (promote_dt_rdata(aval$default_dtype, dtype) == dtype) {
      return(invisible(NULL))
    }
    cli_abort(
      c(
        "Cannot bring {what} to data type {.val {target}}.",
        x = "It is an R {aval$r_type}, which is only ever built at a data type of its own category: a double becomes a float, an integer an integer, a logical a {.val bool}.", # nolint
        i = "Write it in the target's category (e.g. {.code 0L} for an integer data type), or convert it with {.fn nv_convert}." # nolint
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
      i = "Convert it explicitly with {.fn nv_convert}."
    ),
    call = NULL
  )
}

# `promote_yield()`: read the target off the inputs that already have a dtype,
# never move them, and let the R values take it -- within their own category.
resolve_yield <- function(args, positions) {
  avals <- lapply(args[positions], to_abstract)
  is_r <- vapply(avals, is_rdata, logical(1L))
  # An error here is about one particular argument, so it says which.
  labels <- vapply(positions, function(i) arg_label(args, i), character(1L))
  settled <- unique(lapply(avals[!is_r], peek_dtype))
  if (length(settled) > 1L) {
    # The inputs cannot be brought together without converting one of them.
    # Realize nothing and let whatever consumes them report the mismatch.
    return(dtypes_none(args))
  }
  if (!length(settled)) {
    # Every input is an R value. They agree only if they are of one storage
    # type; there is no data type for a mix of them to meet at.
    r_types <- vapply(avals[is_r], function(a) a$r_type, character(1L))
    if (length(unique(r_types)) > 1L) {
      described <- sprintf("%s is an R %s", labels[is_r], r_types)
      cli_abort(
        c(
          "The R values here have no data type to agree on.",
          x = "{described} -- these belong to different data type categories.",
          i = "Give one of them a data type with {.fn nv_scalar} or {.fn nv_array}, or use an operation that promotes across categories." # nolint
        ),
        call = NULL
      )
    }
    return(dtypes_at(args, positions, default_dtype_r(r_types[[1L]])))
  }
  target <- settled[[1L]]
  # Each R value must be able to take the target without leaving its category.
  for (j in which(is_r)) {
    r_type <- avals[[j]]$r_type
    if (!rdata_in_category(r_type, target)) {
      cli_abort(
        c(
          "{labels[[j]]} is an R {r_type}, which cannot be used at the {.val {as.character(target)}} data type here.", # nolint
          i = "A literal is only ever built at a data type of its own category: a double becomes a float, an integer an integer, a logical a {.val bool}.", # nolint
          i = "Use an operation that promotes across categories, or convert explicitly with {.fn nv_convert}."
        ),
        call = NULL
      )
    }
  }
  dtypes_at(args, positions, target)
}

# Resolve argument references -- names or positions -- against the call's
# arguments. `NULL` is every argument, which is what a rule with no `only` covers.
rule_positions <- function(ref, args, what) {
  if (is.null(ref)) {
    return(seq_along(args))
  }
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
common_dtype_of <- function(..., .fallback = NULL) {
  args <- list(...)
  if (length(args) == 0L) {
    cli_abort("No arguments provided")
  }
  cdt <- NULL
  cdt_is_rdata <- TRUE
  for (arg in args) {
    aval <- to_abstract(arg)
    arg_is_rdata <- is_rdata(aval)
    dt <- peek_dtype(aval)
    if (is.null(cdt)) {
      cdt <- dt
      cdt_is_rdata <- arg_is_rdata
      next
    }
    if (cdt_is_rdata && arg_is_rdata) {
      cdt <- promote_dt_known(cdt, dt)
    } else if (cdt_is_rdata) {
      cdt <- promote_dt_rdata(cdt, dt)
      cdt_is_rdata <- FALSE
    } else if (arg_is_rdata) {
      cdt <- promote_dt_rdata(dt, cdt)
    } else {
      cdt <- promote_dt_known(cdt, dt)
    }
  }
  # `.fallback` is the data type the R values commit to when nothing in the call
  # has one of its own to give them -- it replaces the default they would
  # otherwise take, and is ignored the moment any argument brings a real data
  # type. They yield to it as they would to any data type, within their own
  # category, so a fallback below them leaves them where they are.
  if (!is.null(.fallback) && cdt_is_rdata) {
    return(promote_dt_rdata(cdt, as_dtype(.fallback)))
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
