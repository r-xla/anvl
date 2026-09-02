#' @title Type Promotion Rules
#' @description
#' Compute the common data type.
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
#' @name promotion_rule
#' @description
#' Functions for materializing R values to arrays and promoting inputs.
#' Most commonly used via the `.promote` argument of [`as_anvl_arrays()`].
#' @param on (`NULL` | `character()` | `numeric()`)\cr
#'   Subset of arguments to apply a rule to. Indicated either via position or argument name.
#' @param coerce (`logical(1)`)\cr
#'   Bring an input to the target even where that is not a promotion, instead of
#'   raising an error. Two things are refused without it: a float reaching an
#'   integer data type, which no category crosses to on its own (an R double at
#'   `i32`, or an `f32` array at `i32`), and narrowing a value the target cannot
#'   hold (an `f64` array at `f32`). The default is `FALSE`.
#'
#' @return `function(args) -> list()`
#'   A function returning data types for those inputs to be converted and `NULL` for those
#'   to be left unchanged.
#' @seealso [as_anvl_arrays()], [nv_promote_to_common()], [common_dtype()]
NULL

#' @description
#' `promote_common()` brings every input to their common data type
#' ([`common_dtype()`]).
#' R values always yield within the type category (such as float) and otherwise
#' contribute their default data type.
#' @param fallback (`NULL` | [`tengen::DataType`] | `character(1)`)\cr
#'   The data type to settle on when *every* input is a bare R value, in place
#'   of the default those would commit to on their own. `NULL` (default) leaves
#'   them their default.
#' @rdname promotion_rule
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' promote_common()(list(pi, nv_scalar(2L, "i64")))
#' promote_common(fallback = "f64")(list(1, 2))
#' promote_common(c(1, 2))(list(-3, 4, 1))
promote_common <- function(on = NULL, fallback = NULL) {
  assert_on(on)
  fallback <- if (!is.null(fallback)) as_dtype(fallback)
  promotion_rule(
    function(args) {
      positions <- rule_positions(on, args, "on")
      if (!length(positions)) {
        return(dtypes_none(args))
      }
      dtype <- do.call(common_dtype_of, c(args[positions], list(.fallback = fallback)))
      dtypes_at(args, positions, dtype)
    },
    "common",
    on = on,
    fallback = fallback
  )
}

#' @description
#' `promote_like()` brings the inputs to the data type of a selected input.
#' If the selected data type is an R value, it's default data type is used.
#' @param arg (`character(1)` | `numeric(1)`)\cr
#'   Which input to take the data type from: its name in the
#'   [`as_anvl_arrays()`] call, or its position. Naming it needs the call's
#'   arguments to be named.
#' @rdname promotion_rule
#' @export
#' @examplesIf pjrt::plugins_downloaded()
#' promote_like("x", coerce = TRUE)(list(x = nv_scalar(1, "f32"), nv_scalar(1, "f64")))
#' print(promote_like("x")(list(x = nv_scalar(1, "f32"), nv_scalar(1, "f64")), silent = TRUE))
promote_like <- function(arg, on = NULL, coerce = FALSE) {
  assert_arg_ref(arg, "arg", len = 1L)
  assert_on(on)
  assert_flag(coerce)
  promotion_rule(
    function(args) {
      dtype <- do.call(common_dtype_of, args[rule_positions(arg, args, "arg")])
      dtypes_named(args, rule_positions(on, args, "on"), dtype, coerce)
    },
    "like",
    on = on,
    arg = arg,
    coerce = coerce
  )
}

#' @description
#' `promote_dtype()` brings the inputs to the specified data type.
#' @param dtype ([`tengen::DataType`] | `character(1)`)\cr
#'   The data type to bring the inputs to.
#' @rdname promotion_rule
#' @export
promote_dtype <- function(dtype, on = NULL, coerce = FALSE) {
  assert_on(on)
  assert_flag(coerce)
  dtype <- as_dtype(dtype)
  promotion_rule(
    function(args) {
      dtypes_named(args, rule_positions(on, args, "on"), dtype, coerce)
    },
    "dtype",
    on = on,
    dtype = dtype,
    coerce = coerce
  )
}

#' @description
#' `promote_rdata_common()` brings the *R values* to the common data type, as long
#' it is within their category (a `double` can e.g. *not* become a float).
#' `AnvlArray` inputs are left as they are and the function throws an error
#' if not all of them have exactly the same data type.
#' This rule is commonly used in primitives expecting homogenous inputs
#' for one or more argument subsets.
#' @rdname promotion_rule
#' @export
promote_rdata_common <- function(on = NULL) {
  assert_on(on)
  promotion_rule(
    function(args) resolve_rdata_common(args, rule_positions(on, args, "on")),
    "rdata_common",
    on = on
  )
}

#' @description
#' `promote_grouped()` applies several rules to disjoint subsets.
#' @param ... ([`PromotionRule`][promotion_rule])\cr
#'   The rules to apply to disjoint argument subsets.
#' @rdname promotion_rule
#' @export
promote_grouped <- function(...) {
  rules <- list(...)
  if (!length(rules) || !all(vapply(rules, is_promotion_rule, logical(1L)))) {
    cli_abort(c(
      "{.fn promote_grouped} takes promotion rules, one per group of arguments.",
      i = "Build them with {.fn promote_common}, {.fn promote_like}, {.fn promote_dtype} or {.fn promote_rdata_common}, or wrap a rule of your own with {.fn promotion_rule}.", # nolint
      i = "A bare function will not do here: a group has to know which arguments each rule covers before it calls any of them, and only a {.cls PromotionRule} says." # nolint
    ))
  }
  assert_disjoint_rules(rules)
  promotion_rule(
    function(args) {
      # Every rule is asked before any of their answers is used, so one group's
      # target never depends on another's having been applied first.
      dtypes_merged(lapply(rules, resolve_promote, args = args), args)
    },
    "grouped",
    # A group covers what its rules cover together, so a group nested in another
    # answers for its coverage the way any other rule does.
    on = rules_coverage(rules),
    rules = rules
  )
}

rules_coverage <- function(rules) {
  covered <- lapply(rules, function(rule) attr(rule, "spec")$on)
  if (any(vapply(covered, is.null, logical(1L)))) {
    return(NULL)
  }
  if (length(unique(vapply(covered, function(x) is.character(x), logical(1L)))) > 1L) {
    return(NULL)
  }
  unlist(covered)
}

assert_disjoint_rules <- function(rules) {
  covered <- lapply(rules, function(rule) attr(rule, "spec")$on)
  undeclared <- vapply(covered, is.null, logical(1L))
  if (any(undeclared) && length(rules) > 1L) {
    cli_abort(c(
      "Every rule in a {.fn promote_grouped} must say which arguments it covers.",
      x = "{cli::qty(sum(undeclared))}Rule{?s} {.val {which(undeclared)}} {?does/do} not name {.arg on}, so {?it covers/they cover} any argument and cannot be shown disjoint from the others.", # nolint
      i = "Name each group with {.arg on}, or use the rule on its own rather than in a group."
    ))
  }
  if (any(undeclared)) {
    return(invisible(NULL))
  }
  for (kind in c("character", "numeric")) {
    refs <- unlist(Filter(function(x) is(x, kind), covered))
    clashing <- unique(refs[duplicated(refs)])
    if (length(clashing)) {
      cli_abort(c(
        "More than one rule in this {.fn promote_grouped} covers the same argument.",
        x = "{cli::qty(length(clashing))}Argument{?s} {.val {clashing}} {?is/are} named by more than one rule.",
        i = "An argument can only be brought to one data type, so the groups must be disjoint."
      ))
    }
  }
  invisible(NULL)
}

#' @description
#' `promotion_rule()` creates a new promotion rule.
#' It takes in [`arrayish`] values and outputs a list of data types, with `NULL` indicating
#' no conversion.
#' @param fn (`function`)\cr
#'   The rule.
#' @param kind (`character(1)`)\cr
#'   What the rule is, for printing: it shows as `<promote_{kind}>`.
#' @examplesIf pjrt::plugins_downloaded()
#' # Every input at the widest float in the call, and never below f32.
#' widest_float <- promotion_rule(
#'   function(args) {
#'     widths <- vapply(args, function(a) {
#'       dt <- peek_dtype(to_abstract(a))
#'       if (is_dtype_float(dt)) dtype_width(dt) else 0L
#'     }, integer(1))
#'     rep(list(as_dtype(paste0("f", max(c(32L, widths))))), length(args))
#'   },
#'   "widest_float"
#' )
#' widest_float
#' as_anvl_arrays(nv_array(1L), 2.5, nv_array(1, dtype = "f64"), .promote = widest_float)
#' @rdname promotion_rule
#' @export
promotion_rule <- function(fn, kind, on = NULL, ...) {
  checkmate::assert_function(fn)
  checkmate::assert_string(kind)
  assert_on(on)
  structure(
    fn,
    class = c("PromotionRule", "function"),
    kind = kind,
    spec = c(list(on = on), list(...))
  )
}

is_promotion_rule <- function(x) {
  inherits(x, "PromotionRule")
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

assert_on <- function(on) {
  if (!is.null(on)) {
    assert_arg_ref(on, "on")
  }
  invisible(on)
}

#' @export
format.PromotionRule <- function(x, ...) {
  kind <- attr(x, "kind")
  spec <- attr(x, "spec")
  detail <- switch(
    kind,
    common = if (is.null(spec$fallback)) "" else sprintf("(fallback %s)", repr(spec$fallback)),
    rdata_common = "",
    like = sprintf("(%s%s)", format_arg_ref(spec$arg), if (isTRUE(spec$coerce)) ", coerce" else ""),
    dtype = sprintf("(%s%s)", repr(spec$dtype), if (isTRUE(spec$coerce)) ", coerce" else ""),
    grouped = sprintf("(%s)", paste(vapply(spec$rules, format_rule, character(1L)), collapse = ", ")),
    ""
  )
  on <- if (is.null(spec$on)) "" else sprintf(" on %s", format_arg_ref(spec$on))
  sprintf("<promote_%s%s%s>", kind, detail, on)
}

format_rule <- function(x) {
  if (inherits(x, "PromotionRule")) format(x) else "<promotion rule>"
}

format_arg_ref <- function(x) {
  if (is.character(x)) {
    return(paste0("\"", x, "\"", collapse = ", "))
  }
  paste(x, collapse = ", ")
}

#' @export
print.PromotionRule <- function(x, ...) {
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
      i = "Build one with {.fn promote_common}, {.fn promote_like}, {.fn promote_dtype} or {.fn promote_rdata_common}, and combine several with {.fn promote_grouped}." # nolint
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

#' @title Bring a Primitive's Operands to One Data Type
#' @description
#' Applies a [promotion rule][promotion_rule] to the operands of a primitive,
#' inside the primitive's own body.
#'
#' A primitive promotes nothing on its own: an R value among its operands would
#' commit to its own default, so whether a call worked would depend on whether
#' the array it met happened to be at that default. A primitive whose operands
#' must *agree* says so with this, on the same list it goes on to hand
#' [`graph_desc_add()`]:
#'
#' ```r
#' function(lhs, rhs) {
#'   operands <- apply_promotion(list(lhs = lhs, rhs = rhs), promote_rdata_common())
#'   graph_desc_add(self, operands, infer_fn = infer_fn)[[1L]]
#' }
#' ```
#'
#' [`promote_rdata_common()`] is the rule for it: an operand that has a data type
#' keeps it, and an R value takes the one the others have, within its own category.
#' That is what makes `prim_mul(x_f64, 2)` work whatever `x`'s data type is,
#' while keeping a primitive from widening the array it was handed -- promotion
#' across categories is the `nv_*` layer's job.
#'
#' @details
#' Pass only the operands that must agree, and name them as the
#' [`graph_desc_add()`] call names them. [`prim_ifelse()`] promotes its two
#' branches and leaves `pred` a `bool`; [`prim_scatter()`] promotes `x` and
#' `update` and leaves the indices alone. A primitive with one arrayish operand,
#' or with deliberately heterogeneous ones ([`prim_sort()`]'s payload,
#' [`prim_while()`]'s loop state), calls this not at all.
#'
#' Call it before the body uses the operands for anything else, so it sees
#' settled data types throughout: [`prim_reduce()`] reads `dtype(init)` to trace
#' its reductor and [`prim_scatter()`] builds its update computation's parameter
#' slots from [`peek_dtype()`], both before recording a call.
#'
#' It is idempotent: once every operand is at the data type the rule names,
#' realizing them again changes nothing.
#'
#' @param operands (`list()`)\cr
#'   The operands, named as the primitive's [`graph_desc_add()`] call names them.
#' @param promote (`function`)\cr
#'   The rule to apply; see [promotion_rule].
#' @return (`list()`)\cr
#'   `operands`, each realized at the data type the rule named for it.
#' @seealso [promotion_rule], [new_primitive()], `vignette("extending_primitive")`
#' @examplesIf pjrt::plugins_downloaded()
#' # An R value takes the data type of the operand it meets.
#' operands <- apply_promotion(list(lhs = nv_scalar(1, "f64"), rhs = 2), promote_rdata_common())
#' dtype(operands$rhs)
#' @export
apply_promotion <- function(operands, promote) {
  # A primitive with no arrayish operands (`prim_fill()`, `prim_iota()`) has
  # nothing for a rule to answer about.
  if (!length(operands)) {
    return(operands)
  }
  # Realize every operand the rule places at the data type it named, and leave
  # the rest as they are. `as_anvl_arrays()` does the same but converts the
  # untouched ones as well, and places them on a device.
  dtypes <- resolve_promote(promote, operands)
  for (i in seq_along(operands)) {
    if (!is.null(dtypes[[i]])) {
      operands[[i]] <- realize_at(operands[[i]], dtype = dtypes[[i]])
    }
  }
  operands
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
dtypes_named <- function(args, positions, dtype, coerce) {
  if (!isTRUE(coerce)) {
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
        i = "An argument can only be brought to one data type, so the rules must name disjoint sets with {.arg on}." # nolint
      ))
    }
    placed <- c(placed, at)
    out[at] <- dtypes[at]
  }
  out
}

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

# `promote_rdata_common()`: the common data type of inputs that may not be
# converted is the one data type they already share, and the R values take it --
# within their own category.
resolve_rdata_common <- function(args, positions) {
  avals <- lapply(args[positions], to_abstract)
  is_r <- vapply(avals, is_rdata, logical(1L))
  # An error here is about one particular argument, so it says which.
  labels <- vapply(positions, function(i) arg_label(args, i), character(1L))
  settled <- unique(lapply(avals[!is_r], peek_dtype))
  if (length(settled) > 1L) {
    # There is no common data type to reach without converting one of them,
    # which this rule does not do.
    described <- sprintf(
      "%s is %s",
      labels[!is_r],
      vapply(avals[!is_r], function(a) sprintf("`%s`", as.character(peek_dtype(a))), character(1L))
    )
    cli_abort(
      c(
        "These inputs have no common data type to reach without converting one of them.",
        x = "{described}.",
        i = "Use an operation that promotes across data types, or convert one explicitly with {.fn nv_convert}." # nolint
      ),
      call = NULL
    )
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
# arguments. `NULL` is every argument, which is what a rule with no `on` covers.
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


dtype_category <- function(dtype) {
  if (is_dtype_bool(dtype)) {
    1L
  } else if (is_dtype_float(dtype)) {
    3L
  } else {
    2L
  }
}

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
