#' @include array.R
#' @include default-dtypes.R
NULL

# Type inference for every primitive: given the [`AbstractArray`]s flowing into
# a primitive and its static params, work out the [`AbstractArray`]s flowing
# out, and refuse the call when the arguments cannot produce one.
#
# `graph_desc_add()` calls these as `do.call(infer_fn, c(avals_in, params))`, so
# each function's formals are the primitive's own -- every operand and *every*
# param, including the ones inference has no use for (`prim_dot_general()`'s
# `precision`, `prim_sort()`'s `decreasing`). Each returns a plain `list()` of
# `AbstractArray`, named when the primitive has named outputs.
#
# Those formals are what the caller reads: a rule that refuses reaches them with
# its call rewritten to `prim_*()`, so the name a message reports has to be the
# one that primitive takes. Where an operand or param has no argument of its own
# -- the sub-graph behind `prim_reduce()`'s `reducer` or `prim_while()`'s
# `body`, the operands `prim_sort()` collects in `xs`, the ones
# `prim_dynamic_slice()` takes through `...` -- the message names the argument
# itself rather than letting `caller_arg()` report the formal.
#
# The rules are also where a primitive's arguments are checked. Anything that
# can be decided from the incoming avals and the params belongs here and not in
# the `prim_*()` body: a check in both places gives one mistake two wordings.
# Either place reports the primitive as the call: a primitive names itself on
# the way in (`new_primitive()`), and `trace_fn()` rewrites the call of
# anything raised under it.
#
# What stays in the wrapper is what a rule cannot do. `resolve_axis()` and
# `resolve_axes()` normalize rather than check -- they turn a negative axis into
# a concrete one against `naxes(x)`, and the rule only ever sees the result. A
# param stored in the call may need coercing first (`prim_top_k()`'s `k`), which
# goes through the same `assert_*_param()` helper the rule uses so that there is
# still one wording. Sub-graphs are traced before `graph_desc_add()` is reached,
# so their functions are checked there (`prim_reduce()`'s `reducer`,
# `prim_while()`'s `cond` and `body`). And a guard the wrapper's own next line
# depends on stays put -- `prim_sort()` reads `xs[[1L]]` to resolve `axis`.
#
# Whatever a rule refuses, the message says which argument it is about and what
# that argument was given. For a param that means both halves, always: the name
# the primitive takes it under, and the value the caller passed, reported by
# `value_repr()` (or `params_repr()` for several at once). A primitive takes up
# to a dozen params, so a complaint that names none of them leaves the caller
# guessing.
#
# The rules are anvl's own, in anvl's vocabulary: arrays rather than tensors,
# axes rather than dimensions, `x` rather than `operand`, and axis numbers
# 1-based throughout. The constraint numbers in the comments -- (C1), (I2) --
# refer to the StableHLO specification, which is what these rules implement.

# ---------------------------------------------------------------------------
# Shared checking helpers
# ---------------------------------------------------------------------------

# A caller's value as a message reports it back. What a caller hands a param
# is not bounded -- `prim_fill(1:1000, ...)` is one typo away -- so, unlike
# `format_param()`, which prints params that inference has already accepted,
# this has to cope with anything. A long vector is cut to its first
# `repr_max_entries` entries with the length stated, a long string to
# `repr_max_chars` characters, and a matrix or array reads as the call that
# builds it, shape included. A value that is not a plain atomic vector (a
# function, a list, a factor) is reported by class -- and length, for a list --
# where printing it would be noise rather than the mistake. What is left is
# spelled by `format_param()`, so that a value reads the same in an error as in
# a printed graph. Never report a caller's value with a bare `{x}` or
# `{.val {x}}` unless its length is already checked to be 1.
repr_max_entries <- 8L
repr_max_chars <- 30L

value_repr <- function(x) {
  if (is.null(x) || identical(x, list())) {
    return(format_param(x))
  }
  # An array reads as its array type, not as the class it happens to arrive
  # in: under `jit()` an operand is a `GraphBox`, which is the tracer's
  # business and not something the caller wrote.
  if (is_arrayish(x, convert_ok = FALSE)) {
    return(repr(AbstractArray(dtype = dtype(x), shape = shape(x))))
  }
  if (!is.atomic(x) || is.object(x)) {
    # A length is only worth stating for a plain vector, which a list is.
    if (is.vector(x)) {
      return(cli::format_inline("{.cls {class(x)[1L]}} of length {length(x)}"))
    }
    return(cli::format_inline("{.cls {class(x)[1L]}}"))
  }
  shape <- dim(x)
  x <- as.vector(x)
  n <- length(x)
  if (n <= 1L && is.null(shape)) {
    return(value_entries_repr(x))
  }
  entries <- if (n <= 1L) {
    value_entries_repr(x)
  } else {
    # Only the entries shown are formatted, not every one given.
    shown <- value_entries_repr(x[seq_len(min(n, repr_max_entries))])
    paste0("c(", shown, if (n > repr_max_entries) ", ...", ")")
  }
  # A matrix or array reads as the call that builds it, so that its shape is
  # part of what the caller sees. Its shape states its size already.
  if (length(shape) == 2L) {
    return(sprintf("matrix(%s, nrow = %d, ncol = %d)", entries, shape[[1L]], shape[[2L]]))
  }
  if (!is.null(shape)) {
    return(sprintf("array(%s, dim = %s)", entries, value_repr(shape)))
  }
  if (n > repr_max_entries) {
    return(paste0(entries, " of length ", n))
  }
  entries
}

# The entries of a short atomic vector, comma-separated. A string is cut to
# `repr_max_chars` characters, and a missing one reads `NA` rather than the
# `"NA"` that `format_param()` would quote.
value_entries_repr <- function(x) {
  if (!length(x)) {
    return(format_param(x))
  }
  if (is.character(x)) {
    long <- !is.na(x) & nchar(x) > repr_max_chars
    x[long] <- paste0(substr(x[long], 1L, repr_max_chars - 3L), "...")
  }
  entries <- vapply(
    x,
    function(e) if (is.na(e) && !is.nan(e)) "NA" else format_param(e),
    character(1L),
    USE.NAMES = FALSE
  )
  paste(entries, collapse = ", ")
}

# Several params at once: "`strides` = 1, `x_dilation` = c(1, 2)".
params_repr <- function(parts) {
  paste0(
    vapply(
      names(parts),
      function(nm) cli::format_inline("{.arg {nm}} = {value_repr(parts[[nm]])}"),
      character(1L)
    ),
    collapse = ", "
  )
}

# A whole-number parameter the caller supplied: an integer vector with no
# missing values, of `len` entries when the primitive fixes a length. Every
# rule that goes on to index or compare with such a param runs this first --
# otherwise an `NA` reaches an `if ()` and surfaces as the raw R error
# `missing value where TRUE/FALSE needed` under the primitive's name.
assert_int_param <- function(x, arg, len = NULL, min_len = NULL) {
  # `c()` is `NULL`, and it is how a caller spells an empty set of axes.
  if (is.null(x)) {
    x <- integer()
  }
  # A param the primitive fixes at one entry is spoken of in the singular
  # throughout, so that one mistake does not read as "`k` must be a whole
  # number" in one branch and "`k` must contain whole numbers" in the next.
  one <- identical(len, 1L)
  if (!is.numeric(x) || is.object(x)) {
    cli_abort(c(
      "{.arg {arg}} must be a whole number{if (one) '' else ' vector'}.",
      x = "Got {value_repr(x)}."
    ))
  }
  if (anyNA(x)) {
    cli_abort(c(
      "{.arg {arg}} must not {if (one) 'be a missing value' else 'contain missing values'}.",
      x = "Got {value_repr(x)}."
    ))
  }
  if (any(x != trunc(x))) {
    cli_abort(c(
      "{.arg {arg}} must {if (one) 'be a whole number' else 'contain whole numbers'}.",
      x = "Got {value_repr(x)}."
    ))
  }
  # `as.integer()` below would turn these into `NA` with a coercion warning,
  # and a wrapper that coerces first would hand the rule an `NA` the caller
  # never passed.
  if (any(abs(x) > .Machine$integer.max)) {
    cli_abort(c(
      "{.arg {arg}} must {if (one) 'be a whole number' else 'contain whole numbers'} in the integer range.", # nolint
      x = "Got {value_repr(x)}."
    ))
  }
  if (!is.null(len) && length(x) != len) {
    cli_abort(c(
      "{.arg {arg}} must have {len} entr{cli::qty(len)}{?y/ies}.",
      x = "Got {value_repr(x)}."
    ))
  }
  if (!is.null(min_len) && length(x) < min_len) {
    cli_abort(c(
      "{.arg {arg}} must have at least {min_len} entr{cli::qty(min_len)}{?y/ies}.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(as.integer(x))
}

# A data type the caller named. `as_dtype()` refuses an unknown one in its own
# register ("Unsupported dtype: nope"), without the argument or a bullet.
assert_dtype_param <- function(x, arg) {
  out <- tryCatch(as_dtype(x), error = function(e) NULL)
  if (is.null(out)) {
    cli_abort(c(
      "{.arg {arg}} must name a data type.",
      x = "Got {value_repr(x)}.",
      i = "See {.fn tengen::as_dtype} for the data types anvl knows."
    ))
  }
  out
}

assert_flag_param <- function(x, arg) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    cli_abort(c(
      "{.arg {arg}} must be {.val {TRUE}} or {.val {FALSE}}.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(x)
}

# A param the primitive spells out the alternatives for: `prim_round()`'s
# `method`, `prim_dot_general()`'s `precision`.
assert_choice_param <- function(x, arg, choices) {
  if (!rlang::is_string(x) || !(x %in% choices)) {
    cli_abort(c(
      "{.arg {arg}} must be one of {.or {.val {choices}}}.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(x)
}

# Sizes that become a shape must additionally be non-negative, which `Shape()`
# would otherwise report as a bare "Dimensions must be >= 0".
assert_size_param <- function(x, arg, len = NULL) {
  x <- assert_int_param(x, arg, len = len)
  if (any(x < 0L)) {
    cli_abort(c(
      "{.arg {arg}} must not be negative.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(x)
}

# A result shape a rule computed from the caller's params. The arithmetic that
# produces it runs in double, because params `assert_int_param()` accepted one
# by one still overflow when they are summed -- and an `NA` from that overflow
# reaches an `if ()` as the raw R error `missing value where TRUE/FALSE needed`,
# which is what these rules exist to prevent. An axis that no longer fits an R
# integer is refused here rather than at `Shape()`, where it would read as a
# missing value the caller never passed.
assert_result_shape <- function(shape, what, parts) {
  int_max <- .Machine$integer.max
  bad <- which(shape > int_max)
  if (length(bad)) {
    cli_abort(c(
      "{what} must have at most {.val {int_max}} elements along each axis.",
      x = "{cli::qty(length(bad))}Ax{?is/es} {value_repr(bad)} would end up at {value_repr(shape[bad])}.", # nolint
      i = "Got {params_repr(parts)}."
    ))
  }
  as.integer(shape)
}

# The categories as a message reads them, with the article that fits the first:
# "a float", but "an integer or unsigned integer".
dtype_categories_repr <- function(categories) {
  labels <- unname(c(
    float = "float",
    int = "integer",
    uint = "unsigned integer",
    bool = "boolean"
  )[categories])
  article <- if (grepl("^[aeiou]", labels[[1L]])) "an" else "a"
  paste(article, cli::format_inline("{.or {labels}}"), "data type")
}

# Does a dtype belong to a named category?
dtype_in_category <- function(dt, category) {
  switch(
    category,
    float = is_dtype_float(dt),
    int = is_dtype_int(dt),
    uint = is_dtype_uint(dt),
    bool = is_dtype_bool(dt),
    cli_abort("Unknown dtype category: {.val {category}}")
  )
}

assert_array <- function(x, arg = rlang::caller_arg(x)) {
  if (!inherits(x, "AbstractArray")) {
    cli_abort(c(
      "{.arg {arg}} must be an array.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(NULL)
}

# Check several operands at once. Each one is named for the message: the name
# it arrived under, `.arg[[i]]` when the primitive collects the operands in one
# argument (`prim_sort()`'s `xs`), and `..i` when the primitive takes them
# through `...` itself (`prim_concatenate()`).
assert_arrays <- function(..., .arg = NULL) {
  args <- list(...)
  nms <- names(args) %||% rep("", length(args))
  for (i in seq_along(args)) {
    arg <- if (!is.null(.arg)) {
      sprintf("%s[[%i]]", .arg, i)
    } else if (nzchar(nms[[i]])) {
      nms[[i]]
    } else {
      paste0("..", i)
    }
    assert_array(args[[i]], arg = arg)
  }
  invisible(NULL)
}

# Check an array's dtype against one or more category tokens ("float", "int",
# "uint", "bool"), and optionally its shape or rank.
assert_array_dtype <- function(
  x,
  ...,
  shape = NULL,
  naxes = NULL,
  arg = rlang::caller_arg(x)
) {
  assert_array(x, arg = arg)
  categories <- c(...)

  if (length(categories) > 0L) {
    dt <- dtype(x)
    if (!any(vapply(categories, dtype_in_category, logical(1L), dt = dt))) {
      cli_abort(c(
        "{.arg {arg}} must have {dtype_categories_repr(categories)}.",
        x = "Got {.val {as.character(dt)}}."
      ))
    }
  }

  if (!is.null(shape) && !identical(shape(x), as.integer(shape))) {
    cli_abort(c(
      "{.arg {arg}} must have shape {shape_repr(shape)}.",
      x = "Got {shape_repr(shape(x))}."
    ))
  }
  if (!is.null(naxes) && length(shape(x)) != naxes) {
    cli_abort(c(
      "{.arg {arg}} must have {naxes} {cli::qty(naxes)}ax{?is/es}.",
      x = "Got {shape_repr(shape(x))}."
    ))
  }
  invisible(NULL)
}

assert_same_type <- function(
  x,
  y,
  arg_x = rlang::caller_arg(x),
  arg_y = rlang::caller_arg(y)
) {
  assert_array(x, arg = arg_x)
  assert_array(y, arg = arg_y)
  if (eq_type(x, y)) {
    return(invisible(NULL))
  }
  cli_abort(c(
    "{.arg {arg_x}} and {.arg {arg_y}} must have the same array type.",
    x = "Got {repr(x)} and {repr(y)}."
  ))
}

assert_same_dtype <- function(
  x,
  y,
  arg_x = rlang::caller_arg(x),
  arg_y = rlang::caller_arg(y)
) {
  assert_array(x, arg = arg_x)
  assert_array(y, arg = arg_y)
  if (dtype(x) != dtype(y)) {
    cli_abort(c(
      "{.arg {arg_x}} and {.arg {arg_y}} must have the same data type.",
      x = "Got {.val {as.character(dtype(x))}} and {.val {as.character(dtype(y))}}."
    ))
  }
  invisible(NULL)
}

# Axis indices must lie in `1:rank`. At rank 0 there is no such axis at all,
# which "between 1 and 0" would state as a range the caller could satisfy.
assert_axes_in_range <- function(axes, rank, arg) {
  if (!length(axes)) {
    return(invisible(NULL))
  }
  if (rank == 0L) {
    cli_abort(c(
      "{.arg {arg}} cannot be used, there is no axis to select.",
      x = "Got {value_repr(axes)}."
    ))
  }
  if (any(axes < 1L) || any(axes > rank)) {
    cli_abort(c(
      "{.arg {arg}} must contain axes between {.val {1L}} and {.val {rank}}.",
      x = "Got {value_repr(axes)}."
    ))
  }
  invisible(NULL)
}

assert_axes_unique <- function(axes, arg) {
  if (anyDuplicated(axes)) {
    cli_abort(c(
      "{.arg {arg}} must contain unique axes.",
      x = "Got {value_repr(axes)}."
    ))
  }
  invisible(NULL)
}

# A layout: several arguments that between them name every axis of one array
# exactly once (`prim_convolution()`'s three). On a clash the message names the
# arguments that collide, since one of those is what the caller has to change.
assert_axis_layout <- function(parts, rank, what) {
  for (nm in names(parts)) {
    parts[[nm]] <- assert_int_param(parts[[nm]], nm)
  }
  axes <- unlist(parts, use.names = FALSE)
  dup <- axes[duplicated(axes)]
  if (length(dup)) {
    dup <- dup[[1L]]
    n <- sum(axes == dup)
    holders <- names(parts)[vapply(parts, function(p) dup %in% p, logical(1L))]
    cli_abort(c(
      "The axes of {what} must each be named exactly once.",
      x = "Axis {dup} is named {n} times, by {.arg {holders}}.",
      i = "Got {params_repr(parts)}."
    ))
  }
  if (length(axes) != rank) {
    cli_abort(c(
      "The axes of {what} must each be named exactly once.",
      x = "{.arg {names(parts)}} name {cli::qty(length(axes))}{length(axes)} ax{?is/es} between them, but {what} has {rank}.", # nolint
      i = "Got {params_repr(parts)}."
    ))
  }
  for (nm in names(parts)) {
    assert_axes_in_range(parts[[nm]], rank, nm)
  }
  invisible(NULL)
}

assert_axes_sorted <- function(axes, arg) {
  if (is.unsorted(axes)) {
    cli_abort(c(
      "{.arg {arg}} must be sorted in ascending order.",
      x = "Got {value_repr(axes)}."
    ))
  }
  invisible(NULL)
}

# `x` is a permutation of `expected` -- which `setequal()` is not, since sets
# ignore multiplicity and length, so `c(1, 2, 2)` would pass for rank 2.
test_permutation <- function(x, expected) {
  length(x) == length(expected) && setequal(x, expected) && !anyDuplicated(x)
}

# ---------------------------------------------------------------------------
# Element-wise rules
#
# These cover every primitive whose output type is its input type, which is
# most of them. `prim_add()` and friends reach them through `make_binary_op()`
# / `make_unary_op()` in primitives.R.
# ---------------------------------------------------------------------------

# `arg_lhs` / `arg_rhs` are the names the primitive gives its two operands,
# which `make_binary_op()` passes along so a message names the argument the
# caller wrote: `x` / `y` for `prim_pow()`, `x` / `shift` for the shifts.
infer_generic_biv <- function(lhs, rhs, arg_lhs = "lhs", arg_rhs = "rhs") {
  assert_same_type(lhs, rhs, arg_x = arg_lhs, arg_y = arg_rhs)
  list(lhs)
}

infer_numeric_biv <- function(lhs, rhs, arg_lhs = "lhs", arg_rhs = "rhs") {
  assert_array_dtype(lhs, "float", "int", "uint", arg = arg_lhs)
  assert_same_type(lhs, rhs, arg_x = arg_lhs, arg_y = arg_rhs)
  list(lhs)
}

infer_float_biv <- function(lhs, rhs, arg_lhs = "lhs", arg_rhs = "rhs") {
  assert_same_type(lhs, rhs, arg_x = arg_lhs, arg_y = arg_rhs)
  assert_array_dtype(lhs, "float", arg = arg_lhs)
  list(lhs)
}

infer_integerish_biv <- function(lhs, rhs, arg_lhs = "lhs", arg_rhs = "rhs") {
  assert_array_dtype(lhs, "bool", "int", "uint", arg = arg_lhs)
  assert_array_dtype(rhs, "bool", "int", "uint", arg = arg_rhs)
  assert_same_type(lhs, rhs, arg_x = arg_lhs, arg_y = arg_rhs)
  list(lhs)
}

# The bit shifts take a `tensor of integer type`, which in the StableHLO spec
# does not include `i1` -- unlike the bitwise `and` / `or` / `xor` above.
infer_integer_biv <- function(lhs, rhs, arg_lhs = "lhs", arg_rhs = "rhs") {
  assert_array_dtype(lhs, "int", "uint", arg = arg_lhs)
  assert_array_dtype(rhs, "int", "uint", arg = arg_rhs)
  assert_same_type(lhs, rhs, arg_x = arg_lhs, arg_y = arg_rhs)
  list(lhs)
}

infer_generic_uni <- function(x) {
  assert_array(x)
  list(x)
}

infer_numeric_uni <- function(x) {
  assert_array_dtype(x, "float", "int", "uint")
  list(x)
}

infer_float_uni <- function(x) {
  assert_array_dtype(x, "float")
  list(x)
}

infer_integer_uni <- function(x) {
  assert_array_dtype(x, "int", "uint")
  list(x)
}

infer_integerish_uni <- function(x) {
  assert_array_dtype(x, "bool", "int", "uint")
  list(x)
}

# `abs` is numeric-unary but for unsigned input, which it would leave unchanged.
infer_abs <- function(x) {
  assert_array_dtype(x, "float", "int")
  list(x)
}

infer_sign <- function(x) {
  assert_array_dtype(x, "float", "int")
  list(x)
}

infer_is_finite <- function(x) {
  assert_array_dtype(x, "float")
  list(AbstractArray(dtype = "bool", shape = x$shape))
}

infer_polygamma <- function(x, deriv) {
  assert_array_dtype(x, "float")
  assert_array_dtype(deriv, "float")
  assert_same_type(x, deriv)
  list(AbstractArray(dtype = dtype(x), shape = x$shape))
}

# Both rounding methods share an inference rule.
infer_round <- function(x, method) {
  assert_choice_param(method, "method", c("nearest_even", "afz"))
  infer_float_uni(x)
}

# ---------------------------------------------------------------------------
# Shape-transforming rules
# ---------------------------------------------------------------------------

infer_compare <- function(lhs, rhs) {
  assert_same_type(lhs, rhs)
  list(AbstractArray(dtype = "bool", shape = lhs$shape))
}

infer_convert <- function(x, dtype) {
  assert_array(x)
  list(AbstractArray(dtype = assert_dtype_param(dtype, "dtype"), shape = x$shape))
}

infer_bitcast_convert <- function(x, dtype) {
  assert_array(x)
  in_dtype <- dtype(x)
  out_dtype <- assert_dtype_param(dtype, "dtype")

  # https://github.com/openxla/stablehlo/issues/1672
  if (is_dtype_bool(in_dtype)) {
    cli_abort(c(
      "Bitcast conversions from and to {.val bool} are not supported.",
      x = "{.arg x} is {.val {as.character(in_dtype)}}."
    ))
  }
  if (is_dtype_bool(out_dtype)) {
    cli_abort(c(
      "Bitcast conversions from and to {.val bool} are not supported.",
      x = "{.arg dtype} is {.val {as.character(out_dtype)}}."
    ))
  }

  in_width <- dtype_width(in_dtype)
  out_width <- dtype_width(out_dtype)
  in_shape <- shape(x)

  # (C1), (C2) The element widths decide whether an axis is added, dropped or
  # left alone: a wider target packs several input elements into one, so the
  # leading axis it packs along disappears; a narrower one unpacks into a new
  # leading axis of `in_width / out_width`. StableHLO puts that axis last; anvl
  # puts it first, so the pieces of one element are adjacent in column-major
  # order.
  if (in_width == out_width) {
    result_shape <- in_shape
  } else if (in_width < out_width) {
    ratio <- out_width %/% in_width
    if (length(in_shape) == 0L || in_shape[[1L]] != ratio) {
      cli_abort(c(
        "Converting {.val {as.character(in_dtype)}} to the wider {.val {as.character(out_dtype)}} needs a leading axis of {.val {ratio}} to pack.", # nolint
        x = "{.arg x} has shape {shape_repr(in_shape)}."
      ))
    }
    result_shape <- in_shape[-1L]
  } else {
    ratio <- in_width %/% out_width
    result_shape <- c(ratio, in_shape)
  }

  list(AbstractArray(dtype = out_dtype, shape = Shape(result_shape)))
}

infer_broadcast_in_axes <- function(x, shape, broadcast_axes) {
  assert_array(x)
  shape <- assert_shapevec(shape)

  in_shape <- shape(x)
  baxes <- assert_int_param(broadcast_axes, "broadcast_axes")

  # (C2)
  if (length(baxes) != length(in_shape)) {
    cli_abort(c(
      "{.arg broadcast_axes} must have one entry per axis of {.arg x}.",
      x = "Got {value_repr(baxes)} for an {.arg x} with {cli::qty(length(in_shape))}{length(in_shape)} ax{?is/es}." # nolint
    ))
  }

  # (C3), (C4)
  assert_axes_in_range(baxes, length(shape), "broadcast_axes")
  assert_axes_unique(baxes, "broadcast_axes")

  # (C5) Each axis of `x` is either already the size it broadcasts to, or 1.
  for (d in seq_along(baxes)) {
    from <- in_shape[[d]]
    to <- shape[[baxes[[d]]]]
    if (from != to && from != 1L) {
      cli_abort(c(
        "Axis {d} of {.arg x} must be {.or {.val {unique(c(to, 1L))}}} to broadcast to axis {baxes[[d]]} of the result.", # nolint
        x = "Got shapes {shape_repr(in_shape)} and {shape_repr(shape)}."
      ))
    }
  }

  # (C1)
  list(AbstractArray(dtype = dtype(x), shape = Shape(shape)))
}

infer_transpose <- function(x, perm) {
  assert_array(x)
  in_shape <- shape(x)
  perm <- assert_int_param(perm, "perm")

  # (C2) A permutation, which `setequal()` does not check: sets ignore
  # multiplicity and length, so `c(1, 2, 2)` on a rank-2 `x` would pass and
  # (C3) would build a rank-3 result out of it.
  if (!test_permutation(perm, seq_along(in_shape))) {
    if (!length(in_shape)) {
      cli_abort(c(
        "{.arg perm} must be empty, because {.arg x} is a scalar.",
        x = "Got {value_repr(perm)}."
      ))
    }
    cli_abort(c(
      "{.arg perm} must be a permutation of {value_repr(seq_along(in_shape))}.",
      x = "Got {value_repr(perm)}."
    ))
  }

  # (C1), (C3)
  list(AbstractArray(dtype = dtype(x), shape = Shape(in_shape[perm])))
}

infer_reshape <- function(x, shape) {
  assert_array(x)
  result_shape <- assert_size_param(shape, "shape")

  # (C2)
  if (prod(shape(x)) != prod(result_shape)) {
    cli_abort(c(
      "{.arg shape} must have as many elements as {.arg x}.",
      x = "Got {shape_repr(shape(x))} and {shape_repr(result_shape)}."
    ))
  }

  # (C1)
  list(AbstractArray(dtype = dtype(x), shape = Shape(result_shape)))
}

infer_concatenate <- function(..., axis) {
  xs <- list(...)

  # (C3)
  if (!length(xs)) {
    cli_abort("{.arg ...} must hold at least one array to concatenate.")
  }
  assert_arrays(...)

  shapes <- lapply(xs, shape)

  # (C1)
  dtypes <- lapply(xs, dtype)
  if (length(unique(dtypes)) != 1L) {
    bad <- which(vapply(dtypes, function(dt) dt != dtypes[[1L]], logical(1L)))[[1L]]
    bad_arg <- paste0("..", bad)
    cli_abort(c(
      "Every input must have the same data type.",
      x = "{.arg ..1} is {.val {as.character(dtypes[[1L]])}}, {.arg {bad_arg}} is {.val {as.character(dtypes[[bad]])}}." # nolint
    ))
  }

  # (C4)
  rank <- length(shapes[[1L]])
  axis <- assert_int_param(axis, "axis", len = 1L)
  assert_axes_in_range(axis, rank, "axis")

  # (C2) Ranks first, and said separately: "the same shape except along `axis`"
  # is a claim about arrays that have the same axes to begin with. It is also
  # what keeps (C6) in bounds -- `s[-axis]` drops nothing from a shape with
  # fewer axes than `axis`, so `(2x3x4, 2x3)` along axis 3 would otherwise pass
  # here and index past the end of the second shape below.
  ranks <- lengths(shapes)
  if (any(ranks != rank)) {
    bad <- which(ranks != rank)[[1L]]
    bad_arg <- paste0("..", bad)
    cli_abort(c(
      "Every input must have the same number of axes.",
      x = "{.arg ..1} has {cli::qty(rank)}{rank} ax{?is/es} {shape_repr(shapes[[1L]])}, {.arg {bad_arg}} has {ranks[[bad]]} {shape_repr(shapes[[bad]])}." # nolint
    ))
  }

  others <- lapply(shapes, function(s) s[-axis])
  bad <- which(!vapply(others, identical, logical(1L), others[[1L]]))
  if (length(bad)) {
    bad <- bad[[1L]]
    bad_arg <- paste0("..", bad)
    cli_abort(c(
      "Every input must have the same shape except along {.arg axis} ({axis}).",
      x = "{.arg ..1} has shape {shape_repr(shapes[[1L]])}, {.arg {bad_arg}} has shape {shape_repr(shapes[[bad]])}." # nolint
    ))
  }

  # (C6)
  result_shape <- shapes[[1L]]
  result_shape[axis] <- sum(vapply(shapes, function(s) s[[axis]], integer(1L)))

  list(AbstractArray(dtype = dtypes[[1L]], shape = Shape(result_shape)))
}

infer_reverse <- function(x, axes) {
  assert_array(x)
  axes <- assert_int_param(axes, "axes")

  # (C2), (C3). An empty `axes` satisfies both vacuously and StableHLO accepts
  # the program, so it is not refused here -- a lowering that computes the set
  # may legitimately end up with none.
  assert_axes_unique(axes, "axes")
  assert_axes_in_range(axes, length(shape(x)), "axes")

  list(AbstractArray(dtype = dtype(x), shape = x$shape))
}

# `start_indices` and `end_indices` are 1-based and inclusive of the start,
# exclusive of nothing: `end_indices` is the last index kept.
infer_static_slice <- function(x, start_indices, end_indices, strides) {
  assert_array(x)
  start <- assert_int_param(start_indices, "start_indices")
  end <- assert_int_param(end_indices, "end_indices")
  stride <- assert_int_param(strides, "strides")
  in_shape <- shape(x)
  rank <- length(in_shape)

  # (C2)
  given <- list(start_indices = start, end_indices = end, strides = stride)
  if (any(lengths(given) != rank)) {
    cli_abort(c(
      "{.arg start_indices}, {.arg end_indices} and {.arg strides} must have one entry per axis of {.arg x} ({rank}).", # nolint
      x = "Got {params_repr(given)}."
    ))
  }

  # (C3)
  if (any(start < 1L)) {
    bad <- which(start < 1L)
    cli_abort(c(
      "{.arg start_indices} must be at least {.val {1L}}.",
      x = "Got {value_repr(start[bad])} at {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)}."
    ))
  }
  # Before the comparison against `end` below, which would otherwise compute
  # `end + 1L` on an `end` of `.Machine$integer.max` and hand `if ()` the
  # `NA` that overflows to. Once `end` is at most an axis size, it cannot.
  # It is also the better complaint: an out-of-range `end` is the mistake.
  if (any(end > in_shape)) {
    bad <- which(end > in_shape)
    cli_abort(c(
      "{.arg end_indices} must not exceed the shape of {.arg x} {shape_repr(in_shape)}.",
      x = "Got {value_repr(end[bad])} at {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)}."
    ))
  }
  if (any(start > end + 1L)) {
    bad <- which(start > end + 1L)
    cli_abort(c(
      "{.arg start_indices} must not exceed {.arg end_indices}.",
      x = "Got {value_repr(start[bad])} and {value_repr(end[bad])} at {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)}."
    ))
  }

  # (C4) A stride of 0 is not a degenerate slice, it makes the result shape
  # infinite, and MLIR rejects it much later with a raw message plus an R
  # coercion warning from `ceiling(x / 0)`.
  if (any(stride < 1L)) {
    cli_abort(c(
      "{.arg strides} must be positive.",
      x = "Got {value_repr(stride)}."
    ))
  }

  # (C5)
  result_shape <- ceiling((end - start + 1L) / stride)

  list(AbstractArray(dtype = dtype(x), shape = Shape(result_shape)))
}

infer_pad <- function(
  x,
  padding_value,
  edge_padding_low,
  edge_padding_high,
  interior_padding
) {
  assert_arrays(x = x, padding_value = padding_value)

  # (I2)
  if (length(shape(padding_value)) != 0L) {
    cli_abort(c(
      "{.arg padding_value} must be a scalar.",
      x = "Got shape {shape_repr(shape(padding_value))}."
    ))
  }

  # (C1)
  assert_same_dtype(x, padding_value)

  in_shape <- shape(x)
  rank <- length(in_shape)

  low <- assert_int_param(edge_padding_low, "edge_padding_low")
  high <- assert_int_param(edge_padding_high, "edge_padding_high")
  interior <- assert_int_param(interior_padding, "interior_padding")

  # (C2)
  for (nm in c("edge_padding_low", "edge_padding_high", "interior_padding")) {
    val <- switch(nm, edge_padding_low = low, edge_padding_high = high, interior)
    if (length(val) != rank) {
      cli_abort(c(
        "{.arg {nm}} must have one entry per axis of {.arg x} ({rank}).",
        x = "Got {value_repr(val)}."
      ))
    }
  }

  # (C3)
  if (any(interior < 0L)) {
    cli_abort(c(
      "{.arg interior_padding} must be non-negative.",
      x = "Got {value_repr(interior)}."
    ))
  }

  # (C4) Negative edge padding removes elements, and may not remove more than
  # the axis holds.
  result_shape <- in_shape + low + pmax(in_shape - 1L, 0L) * interior + high
  if (any(result_shape < 0L)) {
    bad <- which(result_shape < 0L)
    cli_abort(c(
      "Negative padding must not remove more than an axis holds.",
      x = "{.arg x} has shape {shape_repr(in_shape)}; {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)} would end up at {value_repr(result_shape[bad])}.", # nolint
      i = "Got {params_repr(list(edge_padding_low = low, edge_padding_high = high, interior_padding = interior))}." # nolint
    ))
  }

  list(AbstractArray(dtype = dtype(x), shape = Shape(result_shape)))
}

infer_select <- function(test, yes, no) {
  # (C2)
  assert_same_type(yes, no)
  assert_array_dtype(test, "bool")

  # (C1) `test` selects either element-wise or as a single scalar switch.
  if (length(shape(test)) != 0L && !identical(shape(test), shape(yes))) {
    if (length(shape(yes)) == 0L) {
      cli_abort(c(
        "{.arg yes} and {.arg no} must have {.arg test}'s shape when {.arg test} is not a scalar.",
        x = "{.arg test} is {shape_repr(shape(test))} but the branches are scalars."
      ))
    }
    cli_abort(c(
      "{.arg test} must be a scalar or have the same shape as {.arg yes}.",
      x = "Got {shape_repr(shape(test))} and {shape_repr(shape(yes))}."
    ))
  }

  list(AbstractArray(dtype = dtype(yes), shape = yes$shape))
}

infer_clamp <- function(x, min, max) {
  assert_arrays(x = x, min = min, max = max)

  # (C3)
  assert_same_dtype(x, max, arg_x = "x", arg_y = "max")
  assert_same_dtype(min, x, arg_x = "min", arg_y = "x")

  # (C1) Each bound is either a scalar or the same shape as `x`.
  for (nm in c("min", "max")) {
    bound <- if (nm == "min") min else max
    bound_shape <- shape(bound)
    if (length(bound_shape) != 0L && !identical(bound_shape, shape(x))) {
      cli_abort(c(
        "{.arg {nm}} must be a scalar or have the same shape as {.arg x}.",
        x = "Got {shape_repr(bound_shape)} and {shape_repr(shape(x))}."
      ))
    }
  }

  # (C4)
  list(x)
}

infer_iota <- function(axis, dtype, shape, start) {
  shape <- assert_shapevec(shape)
  axis <- assert_int_param(axis, "axis", len = 1L)
  start <- assert_int_param(start, "start", len = 1L)

  # (C1)
  assert_axes_in_range(axis, length(shape), "axis")

  dtype <- assert_dtype_param(dtype, "dtype")
  if (!any(vapply(c("int", "uint", "float"), dtype_in_category, logical(1L), dt = dtype))) {
    cli_abort(c(
      "{.arg dtype} must name {dtype_categories_repr(c('int', 'uint', 'float'))}.",
      x = "Got {.val {as.character(dtype)}}."
    ))
  }

  list(IotaArray(shape = shape, dtype = dtype, axis = axis, start = start))
}

infer_fill <- function(value, shape, dtype) {
  # `value` is the R scalar the array is filled with, not an operand.
  if (inherits(value, "AbstractArray") || length(value) != 1L) {
    cli_abort(c(
      "{.arg value} must be a scalar.",
      x = "Got {value_repr(value)}."
    ))
  }
  list(AbstractArray(
    dtype = assert_dtype_param(dtype, "dtype"),
    shape = Shape(assert_shapevec(shape))
  ))
}

# `prim_print()` hands its operand straight back, so inserting one cannot
# change what the program computes.
infer_identity <- function(x, ...) {
  list(x)
}

# ---------------------------------------------------------------------------
# Contraction, indexing and reduction rules
# ---------------------------------------------------------------------------

infer_dot_general <- function(
  lhs,
  rhs,
  contracting_axes,
  batching_axes,
  precision
) {
  assert_arrays(lhs = lhs, rhs = rhs)
  # (C13)
  assert_same_dtype(lhs, rhs)
  assert_choice_param(precision, "precision", c("default", "high", "highest"))

  shape_lhs <- shape(lhs)
  shape_rhs <- shape(rhs)
  rank_lhs <- length(shape_lhs)
  rank_rhs <- length(shape_rhs)

  for (nm in c("contracting_axes", "batching_axes")) {
    val <- get(nm)
    if (!is.list(val) || length(val) != 2L) {
      cli_abort(c(
        "{.arg {nm}} must be a list of two axis vectors, one for {.arg lhs} and one for {.arg rhs}.",
        x = "Got {value_repr(val)}."
      ))
    }
  }
  lhs_contracting <- assert_int_param(contracting_axes[[1L]], "contracting_axes[[1]]")
  rhs_contracting <- assert_int_param(contracting_axes[[2L]], "contracting_axes[[2]]")
  lhs_batching <- assert_int_param(batching_axes[[1L]], "batching_axes[[1]]")
  rhs_batching <- assert_int_param(batching_axes[[2L]], "batching_axes[[2]]")

  # (C1)
  if (length(lhs_batching) != length(rhs_batching)) {
    cli_abort(c(
      "{.arg batching_axes} must name as many axes of {.arg lhs} as of {.arg rhs}.",
      x = "Got {value_repr(lhs_batching)} and {value_repr(rhs_batching)}."
    ))
  }

  # (C2)
  if (length(lhs_contracting) != length(rhs_contracting)) {
    cli_abort(c(
      "{.arg contracting_axes} must name as many axes of {.arg lhs} as of {.arg rhs}.",
      x = "Got {value_repr(lhs_contracting)} and {value_repr(rhs_contracting)}."
    ))
  }

  # (C3), (C4)
  assert_axes_unique(
    c(lhs_batching, lhs_contracting),
    "batching_axes[[1]] and contracting_axes[[1]]"
  )
  assert_axes_unique(
    c(rhs_batching, rhs_contracting),
    "batching_axes[[2]] and contracting_axes[[2]]"
  )

  # (C5) - (C8)
  assert_axes_in_range(lhs_batching, rank_lhs, "batching_axes[[1]]")
  assert_axes_in_range(lhs_contracting, rank_lhs, "contracting_axes[[1]]")
  assert_axes_in_range(rhs_batching, rank_rhs, "batching_axes[[2]]")
  assert_axes_in_range(rhs_contracting, rank_rhs, "contracting_axes[[2]]")

  size_contract_lhs <- shape_lhs[lhs_contracting]
  size_contract_rhs <- shape_rhs[rhs_contracting]
  size_batch_lhs <- shape_lhs[lhs_batching]
  size_batch_rhs <- shape_rhs[rhs_batching]

  # (C10)
  if (!identical(size_contract_lhs, size_contract_rhs)) {
    cli_abort(c(
      "The contracted axes of {.arg lhs} and {.arg rhs} must have the same sizes.",
      x = "Axes {value_repr(lhs_contracting)} of {shape_repr(shape_lhs)} are {value_repr(size_contract_lhs)}, axes {value_repr(rhs_contracting)} of {shape_repr(shape_rhs)} are {value_repr(size_contract_rhs)}." # nolint
    ))
  }

  # (C9)
  if (!identical(size_batch_lhs, size_batch_rhs)) {
    cli_abort(c(
      "The batching axes of {.arg lhs} and {.arg rhs} must have the same sizes.",
      x = "Axes {value_repr(lhs_batching)} of {shape_repr(shape_lhs)} are {value_repr(size_batch_lhs)}, axes {value_repr(rhs_batching)} of {shape_repr(shape_rhs)} are {value_repr(size_batch_rhs)}." # nolint
    ))
  }

  used_lhs <- c(lhs_contracting, lhs_batching)
  remaining_lhs <- if (length(used_lhs)) shape_lhs[-used_lhs] else shape_lhs
  used_rhs <- c(rhs_contracting, rhs_batching)
  remaining_rhs <- if (length(used_rhs)) shape_rhs[-used_rhs] else shape_rhs

  # (C12)
  result_shape <- c(size_batch_lhs, remaining_lhs, remaining_rhs)

  list(AbstractArray(dtype = dtype(lhs), shape = Shape(result_shape)))
}

# Every start index is a scalar of the same integer type, one per axis.
assert_start_indices <- function(start_indices, rank) {
  # (C2) / (C4)
  if (length(start_indices) != rank) {
    cli_abort(c(
      "{.arg ...} must hold one start index per axis of {.arg x} ({rank}).",
      x = "Got {length(start_indices)}."
    ))
  }
  if (!length(start_indices)) {
    return(invisible(NULL))
  }
  for (i in seq_along(start_indices)) {
    idx <- start_indices[[i]]
    arg <- sprintf("..%d", i)
    assert_array_dtype(idx, "int", "uint", arg = arg)
    if (length(shape(idx)) != 0L) {
      cli_abort(c(
        "Every start index must be a scalar.",
        x = "{.arg {arg}} has shape {shape_repr(shape(idx))}."
      ))
    }
  }
  # (C3) / (C5)
  dtypes <- lapply(start_indices, dtype)
  if (length(unique(dtypes)) != 1L) {
    bad <- which(vapply(dtypes, function(dt) dt != dtypes[[1L]], logical(1L)))[[1L]]
    bad_arg <- sprintf("..%d", bad)
    cli_abort(c(
      "Every start index must have the same data type.",
      x = "{.arg ..1} is {.val {as.character(dtypes[[1L]])}} and {.arg {bad_arg}} is {.val {as.character(dtypes[[bad]])}}." # nolint
    ))
  }
  invisible(NULL)
}

infer_dynamic_slice <- function(x, ..., slice_sizes) {
  assert_array(x)
  start_indices <- list(...)
  in_shape <- shape(x)
  rank <- length(in_shape)
  sizes <- assert_size_param(slice_sizes, "slice_sizes")

  assert_start_indices(start_indices, rank)

  # (C2)
  if (length(sizes) != rank) {
    cli_abort(c(
      "{.arg slice_sizes} must have one entry per axis of {.arg x} ({rank}).",
      x = "Got {value_repr(sizes)}."
    ))
  }

  # (C4)
  if (any(sizes > in_shape)) {
    cli_abort(c(
      "{.arg slice_sizes} must not exceed the shape of {.arg x}.",
      x = "Got {shape_repr(sizes)} and {shape_repr(in_shape)}."
    ))
  }

  # (C1), (C5)
  list(AbstractArray(dtype = dtype(x), shape = Shape(sizes)))
}

infer_dynamic_update_slice <- function(x, update, ...) {
  assert_arrays(x = x, update = update)
  start_indices <- list(...)
  in_shape <- shape(x)
  rank <- length(in_shape)

  assert_start_indices(start_indices, rank)

  # (C2)
  assert_same_dtype(x, update)

  # (C3)
  if (length(shape(update)) != rank) {
    cli_abort(c(
      "{.arg update} must have as many axes as {.arg x} ({rank}).",
      x = "Got {shape_repr(shape(update))}."
    ))
  }

  # (C6)
  if (any(shape(update) > in_shape)) {
    cli_abort(c(
      "{.arg update} must not be larger than {.arg x} along any axis.",
      x = "Got {shape_repr(shape(update))} and {shape_repr(in_shape)}."
    ))
  }

  # (C1)
  list(AbstractArray(dtype = dtype(x), shape = x$shape))
}

infer_top_k <- function(x, k, indices) {
  assert_array_dtype(x, "float", "int", "uint")
  k <- assert_int_param(k, "k", len = 1L)
  assert_flag_param(indices, "indices")

  in_shape <- shape(x)
  rank <- length(in_shape)

  if (rank < 1L) {
    cli_abort(c(
      "{.arg x} must have at least one axis.",
      x = "Got a scalar."
    ))
  }
  if (k < 1L) {
    cli_abort(c(
      "{.arg k} must be positive.",
      x = "Got {.val {k}}."
    ))
  }
  last <- in_shape[[rank]]
  if (k > last) {
    cli_abort(c(
      "{.arg k} must not exceed the size of the last axis of {.arg x} ({last}).",
      x = "Got {.val {k}}."
    ))
  }

  result_shape <- in_shape
  result_shape[[rank]] <- as.integer(k)

  values <- AbstractArray(dtype = dtype(x), shape = Shape(result_shape))
  if (!indices) {
    return(list(values = values))
  }
  # `hlo_top_k` fixes its indices at `i32`; the lowering converts them to the
  # default integer, which is what the caller sees.
  list(
    values = values,
    indices = AbstractArray(dtype = default_int(), shape = Shape(result_shape))
  )
}

# The shape a reduction leaves behind: `axes` dropped, or kept at size 1.
reduced_shape <- function(x, axes, drop) {
  new_shape <- shape(x)
  if (!length(axes)) {
    return(new_shape)
  }
  if (drop) {
    new_shape <- new_shape[-axes]
  } else {
    new_shape[axes] <- 1L
  }
  new_shape
}

infer_reduce <- function(x, init, axes, drop, reducer) {
  assert_arrays(x = x, init = init)

  if (length(shape(init)) != 0L) {
    cli_abort(c(
      "{.arg init} must be a scalar.",
      x = "Got shape {shape_repr(shape(init))}."
    ))
  }

  assert_flag_param(drop, "drop")

  # (C3)
  assert_same_dtype(x, init)

  # (C4), (C5)
  axes <- assert_int_param(axes, "axes")
  assert_axes_in_range(axes, length(shape(x)), "axes")
  assert_axes_unique(axes, "axes")

  # (C6) The reducer is traced against two scalars of `x`'s data type, so what
  # it returns has to be one too: the reduced element is what it returns, and a
  # different data type there would make the inferred output type a lie.
  outputs <- lapply(reducer$outputs, function(out) out$aval)
  if (length(outputs) != 1L) {
    cli_abort(c(
      "{.arg reducer} must return exactly one value.",
      x = "Got {length(outputs)} outputs."
    ))
  }
  out_aval <- outputs[[1L]]
  if (dtype(out_aval) != dtype(x)) {
    cli_abort(c(
      "{.arg reducer} must return a value with the same data type as {.arg x}.",
      x = "{.arg x} is {.val {as.character(dtype(x))}}, but {.arg reducer} returns {.val {as.character(dtype(out_aval))}}." # nolint
    ))
  }
  if (length(shape(out_aval))) {
    cli_abort(c(
      "{.arg reducer} must return a scalar.",
      x = "Got shape {shape_repr(shape(out_aval))}."
    ))
  }

  # (C7)
  list(AbstractArray(dtype = dtype(x), shape = Shape(reduced_shape(x, axes, drop))))
}

# `prim_sum()` / `prim_max()` and friends: the reducer is fixed
# by the primitive, so the output just keeps the input's data type.
infer_reduce_simple <- function(x, axes, drop) {
  assert_array(x)
  assert_flag_param(drop, "drop")
  axes <- assert_int_param(axes, "axes")
  assert_axes_in_range(axes, length(shape(x)), "axes")
  assert_axes_unique(axes, "axes")
  list(AbstractArray(
    dtype = dtype(x),
    shape = Shape(reduced_shape(x, axes, drop))
  ))
}

infer_reduce_boolean <- function(x, axes, drop) {
  assert_array(x)
  assert_flag_param(drop, "drop")
  axes <- assert_int_param(axes, "axes")
  assert_axes_in_range(axes, length(shape(x)), "axes")
  assert_axes_unique(axes, "axes")
  # The output is a `bool` whatever the input is, but the lowering reduces with
  # `or` / `and` over an init value built at `bool`, so a non-boolean operand
  # fails there with `Data types of inputs and init_values must match`. Say so
  # here instead, where the argument still has a name.
  if (!is_dtype_bool(dtype(x))) {
    cli_abort(
      c(
        "{.arg x} must have a boolean data type.",
        x = "Got {.val {as.character(dtype(x))}}.",
        i = "Compare it first, e.g. {.code x != 0L}, or convert it with {.fn nv_convert}."
      ),
      call = NULL
    )
  }
  list(AbstractArray(
    dtype = "bool",
    shape = Shape(reduced_shape(x, axes, drop))
  ))
}

# `prim_cumsum()` and friends: a scan keeps the input's shape and data type.
infer_cum <- function(x, axis) {
  assert_array(x)
  rank <- length(shape(x))
  if (rank == 0L) {
    cli_abort(c(
      "{.arg x} must have at least one axis to accumulate along.",
      x = "Got a scalar."
    ))
  }
  axis <- assert_int_param(axis, "axis", len = 1L)
  assert_axes_in_range(axis, rank, "axis")
  list(AbstractArray(
    dtype = dtype(x),
    shape = Shape(shape(x))
  ))
}

# `prim_cummax()` / `prim_cummin()`: the running extremum and where it sits.
infer_cum_extreme <- function(x, axis) {
  assert_array(x)
  rank <- length(shape(x))
  if (rank == 0L) {
    cli_abort(c(
      "{.arg x} must have at least one axis to accumulate along.",
      x = "Got a scalar."
    ))
  }
  axis <- assert_int_param(axis, "axis", len = 1L)
  assert_axes_in_range(axis, rank, "axis")
  list(
    values = AbstractArray(
      dtype = dtype(x),
      shape = Shape(shape(x))
    ),
    indices = AbstractArray(
      dtype = default_int(),
      shape = Shape(shape(x))
    )
  )
}

# `prim_which_max()` / `prim_which_min()`: `axis` dropped (or kept at size 1), at the
# default integer data type.
infer_arg_extreme <- function(x, axis, drop) {
  assert_array(x)
  assert_flag_param(drop, "drop")
  axis <- assert_int_param(axis, "axis", len = 1L)
  shp <- shape(x)
  assert_axes_in_range(axis, length(shp), "axis")
  # The reduction lowering uses `init_v = +/-Inf` and `init_i = 0`. Reducing
  # along a size-0 axis would silently emit those sentinels (i.e. index 1)
  # rather than failing. The index of an extremum of nothing is undefined, so
  # reject it here at trace time.
  if (shp[axis] == 0L) {
    cli_abort(c(
      "{.arg x} must have elements along the axis this reads.",
      x = "{.arg x} has shape {shape_repr(shp)}; axis {axis} has size 0."
    ))
  }
  list(AbstractArray(
    dtype = default_int(),
    shape = Shape(reduced_shape(x, axis, drop))
  ))
}

# `prim_sort()` only permutes along `axis`, so each output mirrors its input.
# The arrays arrive as separate operands, but the primitive takes them in a
# single `xs`, which is the argument a message has to name.
infer_sort <- function(..., axis, decreasing, stable) {
  xs <- list(...)
  if (!length(xs)) {
    cli_abort(c(
      "{.arg xs} must be a non-empty list of arrayish values.",
      x = "Got nothing to sort."
    ))
  }
  assert_flag_param(decreasing, "decreasing")
  assert_flag_param(stable, "stable")
  assert_arrays(..., .arg = "xs")

  # (C1), (C2) The payloads are permuted alongside the key, so they have to
  # line up with it; only their data types are free to differ.
  ref <- shape(xs[[1L]])
  bad <- which(!vapply(xs, function(x) identical(shape(x), ref), logical(1L)))
  if (length(bad)) {
    cli_abort(c(
      "Every element of {.arg xs} must have the same shape.",
      x = "{.arg xs[[1]]} has shape {shape_repr(ref)}, {.arg xs[[{bad[[1L]]}]]} has shape {shape_repr(shape(xs[[bad[[1L]]]]))}." # nolint
    ))
  }

  axis <- assert_int_param(axis, "axis", len = 1L)
  assert_axes_in_range(axis, length(ref), "axis")
  lapply(xs, function(x) AbstractArray(dtype = dtype(x), shape = x$shape))
}

# ---------------------------------------------------------------------------
# Gather and scatter
#
# `index_vector_axis` may be `rank + 1`, which means the index vectors are
# implicit: each entry of `start_indices` / `scatter_indices` is a single index
# rather than a vector of them.
# ---------------------------------------------------------------------------

infer_gather <- function(
  x,
  start_indices,
  slice_sizes,
  offset_axes,
  collapsed_slice_axes,
  x_batching_axes,
  start_indices_batching_axes,
  start_index_map,
  index_vector_axis,
  indices_are_sorted,
  unique_indices
) {
  assert_array(x)
  # (I2)
  assert_array_dtype(start_indices, "int", "uint")

  x_shape <- shape(x)
  x_rank <- length(x_shape)
  idx_shape <- shape(start_indices)
  idx_rank <- length(idx_shape)
  sizes <- assert_size_param(slice_sizes, "slice_sizes")
  index_vector_axis <- assert_int_param(index_vector_axis, "index_vector_axis", len = 1L)
  assert_flag_param(indices_are_sorted, "indices_are_sorted")
  assert_flag_param(unique_indices, "unique_indices")

  offset_axes <- assert_int_param(offset_axes, "offset_axes")
  collapsed_slice_axes <- assert_int_param(collapsed_slice_axes, "collapsed_slice_axes")
  x_batching_axes <- assert_int_param(x_batching_axes, "x_batching_axes")
  start_indices_batching_axes <- assert_int_param(start_indices_batching_axes, "start_indices_batching_axes")
  start_index_map <- assert_int_param(start_index_map, "start_index_map")

  # (C20) first: `offset_sizes` below is `sizes` with axes removed, so a
  # wrong-length `slice_sizes` would otherwise surface as an `offset_axes`
  # complaint about a result rank computed from it.
  if (length(sizes) != x_rank) {
    cli_abort(c(
      "{.arg slice_sizes} must have one entry per axis of {.arg x} ({x_rank}).",
      x = "Got {value_repr(sizes)}."
    ))
  }

  # (C1) Every axis of `x` is either offset, collapsed or batching.
  expected_rank <- length(offset_axes) +
    length(collapsed_slice_axes) +
    length(x_batching_axes)
  if (x_rank != expected_rank) {
    cli_abort(c(
      "{.arg x} must have one axis per entry of {.arg offset_axes}, {.arg collapsed_slice_axes} and {.arg x_batching_axes}.", # nolint
      x = "{.arg x} has {cli::qty(x_rank)}{x_rank} ax{?is/es}, but those name {expected_rank} ({length(offset_axes)} + {length(collapsed_slice_axes)} + {length(x_batching_axes)})." # nolint
    ))
  }

  # (C2)
  if (index_vector_axis < 1L || index_vector_axis > idx_rank + 1L) {
    cli_abort(c(
      "{.arg index_vector_axis} must be between {.val {1L}} and {.val {idx_rank + 1L}}.",
      x = "Got {.val {index_vector_axis}}."
    ))
  }

  # (C3)
  expected_map_size <- if (index_vector_axis <= idx_rank) {
    idx_shape[[index_vector_axis]]
  } else {
    1L
  }
  if (length(start_index_map) != expected_map_size) {
    cli_abort(c(
      "{.arg start_index_map} must have one entry per index coordinate ({expected_map_size}).",
      x = "Got {value_repr(start_index_map)}."
    ))
  }

  # (C4)
  assert_axes_unique(offset_axes, "offset_axes")
  assert_axes_sorted(offset_axes, "offset_axes")

  batch_sizes <- if (index_vector_axis == idx_rank + 1L) {
    idx_shape
  } else {
    without(idx_shape, index_vector_axis)
  }
  offset_sizes <- without(sizes, c(collapsed_slice_axes, x_batching_axes))
  result_rank <- length(batch_sizes) + length(offset_sizes)

  # (C5)
  assert_axes_in_range(offset_axes, result_rank, "offset_axes")

  # (C6)
  assert_axes_unique(
    c(collapsed_slice_axes, x_batching_axes),
    "collapsed_slice_axes and x_batching_axes"
  )

  # (C7), (C8)
  assert_axes_sorted(collapsed_slice_axes, "collapsed_slice_axes")
  assert_axes_in_range(collapsed_slice_axes, x_rank, "collapsed_slice_axes")

  # (C9) A collapsed axis contributes nothing to the result, so its slice must
  # be a single element.
  if (length(collapsed_slice_axes) && any(sizes[collapsed_slice_axes] > 1L)) {
    bad <- collapsed_slice_axes[sizes[collapsed_slice_axes] > 1L]
    cli_abort(c(
      "{.arg slice_sizes} must be at most {.val {1L}} at {.arg collapsed_slice_axes}.",
      x = "Got {value_repr(sizes[bad])} at {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)}."
    ))
  }

  # (C10), (C11)
  assert_axes_sorted(x_batching_axes, "x_batching_axes")
  assert_axes_in_range(x_batching_axes, x_rank, "x_batching_axes")

  # (C12)
  if (length(x_batching_axes) && any(sizes[x_batching_axes] > 1L)) {
    bad <- x_batching_axes[sizes[x_batching_axes] > 1L]
    cli_abort(c(
      "{.arg slice_sizes} must be at most {.val {1L}} at {.arg x_batching_axes}.",
      x = "Got {value_repr(sizes[bad])} at {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)}."
    ))
  }

  # (C13), (C14)
  assert_axes_unique(start_indices_batching_axes, "start_indices_batching_axes")
  assert_axes_in_range(start_indices_batching_axes, idx_rank, "start_indices_batching_axes")

  # (C15)
  if (index_vector_axis %in% start_indices_batching_axes) {
    cli_abort(c(
      "{.arg index_vector_axis} must not be one of {.arg start_indices_batching_axes}.",
      x = "{.arg index_vector_axis} is {.val {index_vector_axis}} and {.arg start_indices_batching_axes} is {value_repr(start_indices_batching_axes)}." # nolint
    ))
  }

  # (C16)
  if (length(x_batching_axes) != length(start_indices_batching_axes)) {
    cli_abort(c(
      "{.arg x_batching_axes} and {.arg start_indices_batching_axes} must have the same length.",
      x = "Got {value_repr(x_batching_axes)} and {value_repr(start_indices_batching_axes)}."
    ))
  }

  # (C17)
  if (length(x_batching_axes)) {
    batch_x <- x_shape[x_batching_axes]
    batch_idx <- idx_shape[start_indices_batching_axes]
    if (!identical(batch_x, batch_idx)) {
      cli_abort(c(
        "The batching axes of {.arg x} and {.arg start_indices} must have the same sizes.",
        x = "Got {shape_repr(batch_x)} and {shape_repr(batch_idx)}."
      ))
    }
  }

  # (C18), (C19)
  assert_axes_unique(
    c(start_index_map, x_batching_axes),
    "start_index_map and x_batching_axes"
  )
  assert_axes_in_range(start_index_map, x_rank, "start_index_map")

  # (C21)
  if (any(sizes < 0L) || any(sizes > x_shape)) {
    cli_abort(c(
      "{.arg slice_sizes} must be between {.val {0L}} and the shape of {.arg x} {shape_repr(x_shape)}.",
      x = "Got {value_repr(sizes)}."
    ))
  }

  # (C22) The result interleaves the batch axes with the offset axes, with
  # `offset_axes` saying where the latter land.
  batch_axes <- setdiff(seq_len(result_rank), offset_axes)
  result_shape <- integer(result_rank)
  if (length(batch_axes)) {
    result_shape[batch_axes] <- batch_sizes
  }
  if (length(offset_axes)) {
    result_shape[offset_axes] <- offset_sizes
  }

  # (C23)
  list(AbstractArray(dtype = dtype(x), shape = Shape(result_shape)))
}

infer_scatter <- function(
  x,
  scatter_indices,
  update,
  update_window_axes,
  inserted_window_axes,
  x_batching_axes,
  scatter_indices_batching_axes,
  scatter_axes_to_x_axes,
  index_vector_axis,
  indices_are_sorted,
  unique_indices,
  update_fn
) {
  assert_arrays(x = x, update = update)
  # (I2)
  assert_array_dtype(scatter_indices, "int", "uint")

  x_shape <- shape(x)
  x_rank <- length(x_shape)
  idx_shape <- shape(scatter_indices)
  idx_rank <- length(idx_shape)
  update_shape <- shape(update)
  update_rank <- length(update_shape)

  index_vector_axis <- assert_int_param(index_vector_axis, "index_vector_axis", len = 1L)
  assert_flag_param(indices_are_sorted, "indices_are_sorted")
  assert_flag_param(unique_indices, "unique_indices")

  update_window_axes <- assert_int_param(update_window_axes, "update_window_axes")
  inserted_window_axes <- assert_int_param(inserted_window_axes, "inserted_window_axes")
  x_batching_axes <- assert_int_param(x_batching_axes, "x_batching_axes")
  scatter_indices_batching_axes <- assert_int_param(scatter_indices_batching_axes, "scatter_indices_batching_axes")
  scatter_axes_to_x_axes <- assert_int_param(scatter_axes_to_x_axes, "scatter_axes_to_x_axes")

  # (C7) before (C2): a repeated entry makes the count below wrong, and
  # "must contain unique axes" is the complaint the caller can act on.
  assert_axes_unique(update_window_axes, "update_window_axes")

  # (C2)
  expected_rank <- length(update_window_axes) +
    length(inserted_window_axes) +
    length(x_batching_axes)
  if (x_rank != expected_rank) {
    cli_abort(c(
      "{.arg x} must have one axis per entry of {.arg update_window_axes}, {.arg inserted_window_axes} and {.arg x_batching_axes}.", # nolint
      x = "{.arg x} has {cli::qty(x_rank)}{x_rank} ax{?is/es}, but those name {expected_rank} ({length(update_window_axes)} + {length(inserted_window_axes)} + {length(x_batching_axes)})." # nolint
    ))
  }

  # (C6)
  assert_same_dtype(x, update)

  # (C22)
  if (index_vector_axis < 1L || index_vector_axis > idx_rank + 1L) {
    cli_abort(c(
      "{.arg index_vector_axis} must be between {.val {1L}} and {.val {idx_rank + 1L}}.",
      x = "Got {.val {index_vector_axis}}."
    ))
  }

  # (C4) `scatter_indices` gains an implicit trailing axis when the index
  # vectors are implicit, and `update` carries one axis per remaining index
  # axis plus one per window axis.
  expanded_idx_rank <- if (index_vector_axis == idx_rank + 1L) {
    idx_rank + 1L
  } else {
    idx_rank
  }
  expected_update_rank <- expanded_idx_rank - 1L + length(update_window_axes)
  if (update_rank != expected_update_rank) {
    cli_abort(c(
      "{.arg update} must have {cli::qty(expected_update_rank)}{expected_update_rank} ax{?is/es}.",
      x = "Got {shape_repr(update_shape)}."
    ))
  }

  # (C8)
  assert_axes_sorted(update_window_axes, "update_window_axes")
  assert_axes_in_range(update_window_axes, update_rank, "update_window_axes")

  # (C9), (C10), (C11)
  assert_axes_unique(
    c(inserted_window_axes, x_batching_axes),
    "inserted_window_axes and x_batching_axes"
  )
  assert_axes_sorted(inserted_window_axes, "inserted_window_axes")
  assert_axes_in_range(inserted_window_axes, x_rank, "inserted_window_axes")

  # (C12), (C13)
  assert_axes_sorted(x_batching_axes, "x_batching_axes")
  assert_axes_in_range(x_batching_axes, x_rank, "x_batching_axes")

  # (C14), (C15)
  assert_axes_unique(scatter_indices_batching_axes, "scatter_indices_batching_axes")
  assert_axes_in_range(scatter_indices_batching_axes, idx_rank, "scatter_indices_batching_axes")

  # (C16)
  if (index_vector_axis %in% scatter_indices_batching_axes) {
    cli_abort(c(
      "{.arg index_vector_axis} must not be one of {.arg scatter_indices_batching_axes}.",
      x = "{.arg index_vector_axis} is {.val {index_vector_axis}} and {.arg scatter_indices_batching_axes} is {value_repr(scatter_indices_batching_axes)}." # nolint
    ))
  }

  # (C17)
  if (length(x_batching_axes) != length(scatter_indices_batching_axes)) {
    cli_abort(c(
      "{.arg x_batching_axes} and {.arg scatter_indices_batching_axes} must have the same length.",
      x = "Got {value_repr(x_batching_axes)} and {value_repr(scatter_indices_batching_axes)}."
    ))
  }

  # (C18)
  batch_x <- x_shape[x_batching_axes]
  batch_idx <- idx_shape[scatter_indices_batching_axes]
  if (!identical(batch_x, batch_idx)) {
    cli_abort(c(
      "The batching axes of {.arg x} and {.arg scatter_indices} must have the same sizes.",
      x = "Got {shape_repr(batch_x)} and {shape_repr(batch_idx)}."
    ))
  }

  # (C19)
  expected_map_size <- if (index_vector_axis <= idx_rank) {
    idx_shape[[index_vector_axis]]
  } else {
    1L
  }
  if (length(scatter_axes_to_x_axes) != expected_map_size) {
    cli_abort(c(
      "{.arg scatter_axes_to_x_axes} must have one entry per index coordinate ({expected_map_size}).",
      x = "Got {value_repr(scatter_axes_to_x_axes)}."
    ))
  }

  # (C20), (C21)
  assert_axes_unique(
    c(scatter_axes_to_x_axes, x_batching_axes),
    "scatter_axes_to_x_axes and x_batching_axes"
  )
  assert_axes_in_range(scatter_axes_to_x_axes, x_rank, "scatter_axes_to_x_axes")

  update_scatter_axes <- setdiff(seq_len(update_rank), update_window_axes)

  scatter_sizes <- if (index_vector_axis == idx_rank + 1L) {
    idx_shape
  } else {
    without(idx_shape, index_vector_axis)
  }
  window_sizes <- without(x_shape, c(inserted_window_axes, x_batching_axes))

  if (length(update_window_axes)) {
    actual_window <- update_shape[update_window_axes]
    if (any(actual_window > window_sizes)) {
      cli_abort(c(
        "{.arg update} must not be larger than {.arg x} along its window axes.",
        x = "Got {value_repr(actual_window)}, at most {value_repr(window_sizes)} is allowed."
      ))
    }
  }

  if (length(update_scatter_axes)) {
    actual_scatter <- update_shape[update_scatter_axes]
    if (!identical(actual_scatter, scatter_sizes)) {
      cli_abort(c(
        "The scatter axes of {.arg update} must match the shape of {.arg scatter_indices}.",
        x = "Got {value_repr(actual_scatter)}, expected {value_repr(scatter_sizes)}."
      ))
    }
  }

  # (C23) As `prim_reduce()`'s reducer: `update_fn` is traced against
  # two scalars of `x`'s data type, so what it returns has to be one too.
  outputs <- lapply(update_fn$outputs, function(out) out$aval)
  if (length(outputs) != 1L) {
    cli_abort(c(
      "{.arg update_fn} must return exactly one value.",
      x = "Got {length(outputs)} outputs."
    ))
  }
  out_aval <- outputs[[1L]]
  if (dtype(out_aval) != dtype(x)) {
    cli_abort(c(
      "{.arg update_fn} must return a value with the same data type as {.arg x}.",
      x = "{.arg x} is {.val {as.character(dtype(x))}} and {.arg update_fn} returns {.val {as.character(dtype(out_aval))}}." # nolint
    ))
  }

  # (C24), (C25)
  list(AbstractArray(dtype = dtype(x), shape = Shape(x_shape)))
}

# ---------------------------------------------------------------------------
# Convolution and linear algebra
# ---------------------------------------------------------------------------

infer_convolution <- function(
  x,
  kernel,
  x_batch_axis,
  x_feature_axis,
  x_spatial_axes,
  kernel_input_feature_axis,
  kernel_output_feature_axis,
  kernel_spatial_axes,
  output_batch_axis,
  output_feature_axis,
  output_spatial_axes,
  window_strides,
  padding,
  x_dilation,
  kernel_dilation,
  feature_group_count,
  batch_group_count,
  precision
) {
  assert_arrays(x = x, kernel = kernel)

  x_shape <- shape(x)
  kernel_shape <- shape(kernel)
  rank <- length(x_shape)

  # (C1)
  if (rank != length(kernel_shape)) {
    cli_abort(c(
      "{.arg x} and {.arg kernel} must have the same number of axes.",
      x = "Got {shape_repr(x_shape)} and {shape_repr(kernel_shape)}."
    ))
  }
  if (rank < 2L) {
    cli_abort(c(
      "{.arg x} and {.arg kernel} must have at least two axes.",
      x = "Got {shape_repr(x_shape)} and {shape_repr(kernel_shape)}."
    ))
  }
  n_spatial <- rank - 2L

  strides <- assert_int_param(window_strides, "window_strides")
  x_dil <- assert_int_param(x_dilation, "x_dilation")
  kernel_dil <- assert_int_param(kernel_dilation, "kernel_dilation")
  fg_count <- assert_int_param(feature_group_count, "feature_group_count", len = 1L)
  bg_count <- assert_int_param(batch_group_count, "batch_group_count", len = 1L)
  assert_choice_param(precision, "precision", c("default", "high", "highest"))
  pad_dim <- dim(padding)
  if (is.null(pad_dim) || !identical(as.integer(pad_dim), c(n_spatial, 2L))) {
    cli_abort(c(
      "{.arg padding} must be a matrix of shape {shape_repr(c(n_spatial, 2L))}.",
      x = if (is.null(pad_dim)) {
        "Got a vector of length {length(padding)}."
      } else {
        "Got {shape_repr(pad_dim)}."
      }
    ))
  }
  # The only integer parameter here that keeps its shape, so it is checked
  # after the dim test rather than through `assert_int_param()`.
  pad <- matrix(assert_int_param(as.vector(padding), "padding"), nrow = n_spatial, ncol = 2L)

  # (C2) - (C9) Each per-spatial-axis vector has one entry per spatial axis,
  # and the strides and dilations are positive.
  for (nm in c("window_strides", "x_dilation", "kernel_dilation")) {
    val <- switch(nm, window_strides = strides, x_dilation = x_dil, kernel_dil)
    if (length(val) != n_spatial) {
      cli_abort(c(
        "{.arg {nm}} must have one entry per spatial axis ({n_spatial}).",
        x = "Got {value_repr(val)}."
      ))
    }
    if (any(val <= 0L)) {
      cli_abort(c(
        "{.arg {nm}} must be positive.",
        x = "Got {value_repr(val)}."
      ))
    }
  }

  # (C21) - (C23)
  if (fg_count <= 0L) {
    cli_abort(c(
      "{.arg feature_group_count} must be positive.",
      x = "Got {.val {fg_count}}."
    ))
  }
  if (bg_count <= 0L) {
    cli_abort(c(
      "{.arg batch_group_count} must be positive.",
      x = "Got {.val {bg_count}}."
    ))
  }
  if (fg_count != 1L && bg_count != 1L) {
    cli_abort(c(
      "At least one of {.arg feature_group_count} and {.arg batch_group_count} must be {.val {1L}}.",
      x = "Got {fg_count} and {bg_count}."
    ))
  }

  # (C12), (C17), (C19)
  for (nm in c("x_spatial_axes", "kernel_spatial_axes", "output_spatial_axes")) {
    # Asserted here rather than left to `assert_axis_layout()` below, so that a
    # param that is not a whole-number vector is reported as that, and one that
    # is can be reported by its entries.
    val <- assert_int_param(
      switch(
        nm,
        x_spatial_axes = x_spatial_axes,
        kernel_spatial_axes = kernel_spatial_axes,
        output_spatial_axes
      ),
      nm
    )
    if (length(val) != n_spatial) {
      cli_abort(c(
        "{.arg {nm}} must have one entry per spatial axis ({n_spatial}).",
        x = "Got {value_repr(val)}."
      ))
    }
  }

  # (C13), (C18), (C20) Each of the three layouts names every axis exactly once.
  assert_axis_layout(
    list(
      x_batch_axis = x_batch_axis,
      x_spatial_axes = x_spatial_axes,
      x_feature_axis = x_feature_axis
    ),
    rank,
    "x"
  )
  assert_axis_layout(
    list(
      kernel_spatial_axes = kernel_spatial_axes,
      kernel_input_feature_axis = kernel_input_feature_axis,
      kernel_output_feature_axis = kernel_output_feature_axis
    ),
    rank,
    "kernel"
  )
  assert_axis_layout(
    list(
      output_batch_axis = output_batch_axis,
      output_spatial_axes = output_spatial_axes,
      output_feature_axis = output_feature_axis
    ),
    rank,
    "the result"
  )

  x_batch_size <- x_shape[[x_batch_axis]]
  x_feature_size <- x_shape[[x_feature_axis]]
  kernel_in_size <- kernel_shape[[kernel_input_feature_axis]]
  kernel_out_size <- kernel_shape[[kernel_output_feature_axis]]

  # (C10), (C11)
  if (x_batch_size %% bg_count != 0L) {
    cli_abort(c(
      "The batch axis of {.arg x} must be divisible by {.arg batch_group_count}.",
      x = "Got {x_batch_size} and {bg_count}."
    ))
  }
  if (x_feature_size %% fg_count != 0L) {
    cli_abort(c(
      "The feature axis of {.arg x} must be divisible by {.arg feature_group_count}.",
      x = "Got {x_feature_size} and {fg_count}."
    ))
  }

  # (C14) - (C16)
  if (kernel_in_size != x_feature_size %/% fg_count) {
    cli_abort(c(
      "The input feature axis of {.arg kernel} must be the feature axis of {.arg x} divided by {.arg feature_group_count}.", # nolint
      x = "Got {kernel_in_size}, expected {x_feature_size %/% fg_count}."
    ))
  }
  if (kernel_out_size %% bg_count != 0L) {
    cli_abort(c(
      "The output feature axis of {.arg kernel} must be divisible by {.arg batch_group_count}.",
      x = "Got {kernel_out_size} and {bg_count}."
    ))
  }
  if (kernel_out_size %% fg_count != 0L) {
    cli_abort(c(
      "The output feature axis of {.arg kernel} must be divisible by {.arg feature_group_count}.",
      x = "Got {kernel_out_size} and {fg_count}."
    ))
  }

  # (C27)
  assert_same_dtype(x, kernel, arg_x = "x", arg_y = "kernel")

  # (C25), (C26)
  result_shape <- integer(rank)
  result_shape[output_batch_axis] <- x_batch_size %/% bg_count
  result_shape[output_feature_axis] <- kernel_out_size
  # The window arithmetic runs in double: a dilation or a padding in the
  # billions overflows an integer, and the `NA` it produces reaches the tests
  # below as the raw `missing value where TRUE/FALSE needed`.
  result_shape <- as.double(result_shape)
  for (sd in seq_len(n_spatial)) {
    x_size <- as.double(x_shape[[x_spatial_axes[[sd]]]])
    k_size <- as.double(kernel_shape[[kernel_spatial_axes[[sd]]]])
    dilated_input <- if (x_size == 0) 0 else (x_size - 1) * x_dil[[sd]] + 1
    padded_input <- pad[sd, 1L] + dilated_input + pad[sd, 2L]

    # Negative padding may empty a spatial axis but must not take away more
    # than it holds: XLA infers the window bound from this and `CHECK`-fails on
    # a negative one, which aborts the process rather than raising an error, so
    # a negative extent can never leave this rule.
    if (padded_input < 0) {
      cli_abort(c(
        "Negative {.arg padding} must not remove more than spatial axis {sd} of {.arg x} holds.",
        x = "Axis {x_spatial_axes[[sd]]} of {.arg x} dilates to {dilated_input}, and padding {pad[sd, 1L]} and {pad[sd, 2L]} leaves {padded_input}.", # nolint
        i = "Got {params_repr(list(padding = padding, x_dilation = x_dil))}."
      ))
    }

    # A window has to have something in it. A zero-sized kernel axis gives a
    # zero-wide window, which the arithmetic below reads as "not wider than the
    # input" and infers a non-empty result from; StableHLO refuses it outright,
    # reporting the axis 0-based.
    if (k_size == 0) {
      cli_abort(c(
        "{.arg kernel} must not have a zero-sized spatial axis.",
        x = "Axis {kernel_spatial_axes[[sd]]} of {.arg kernel} is {.val {0L}}."
      ))
    }

    dilated_window <- (k_size - 1) * kernel_dil[[sd]] + 1
    num_windows <- if (padded_input == 0 || dilated_window > padded_input) {
      0
    } else {
      floor((padded_input - dilated_window) / strides[[sd]]) + 1
    }
    result_shape[output_spatial_axes[[sd]]] <- num_windows
  }

  result_shape <- assert_result_shape(
    result_shape,
    "The convolution's result",
    list(
      padding = padding,
      window_strides = strides,
      x_dilation = x_dil,
      kernel_dilation = kernel_dil
    )
  )

  list(AbstractArray(dtype = dtype(x), shape = Shape(result_shape)))
}

infer_triangular_solve <- function(a, b, left_side, lower, unit_diagonal, transpose_a) {
  assert_flag_param(left_side, "left_side")
  assert_flag_param(lower, "lower")
  assert_flag_param(unit_diagonal, "unit_diagonal")
  assert_flag_param(transpose_a, "transpose_a")
  # (I1), (I2) Floating-point operands only; complex is not supported yet.
  assert_array_dtype(a, "float")
  assert_array_dtype(b, "float")
  # (C1)
  assert_same_dtype(a, b)

  shape_a <- shape(a)
  shape_b <- shape(b)
  rank_a <- length(shape_a)
  rank_b <- length(shape_b)

  # (C2)
  if (rank_a < 2L) {
    cli_abort(c(
      "{.arg a} must have at least two axes.",
      x = "Got shape {shape_repr(shape_a)}."
    ))
  }
  if (rank_a != rank_b) {
    cli_abort(c(
      "{.arg a} and {.arg b} must have the same number of axes.",
      x = "Got {shape_repr(shape_a)} and {shape_repr(shape_b)}."
    ))
  }

  # (C3)
  if (shape_a[[rank_a]] != shape_a[[rank_a - 1L]]) {
    cli_abort(c(
      "{.arg a} must be square in its last two axes.",
      x = "Got {shape_repr(shape_a)}."
    ))
  }

  if (rank_a > 2L) {
    batch_a <- shape_a[seq_len(rank_a - 2L)]
    batch_b <- shape_b[seq_len(rank_b - 2L)]
    if (!identical(batch_a, batch_b)) {
      cli_abort(c(
        "The batch axes of {.arg a} and {.arg b} must match.",
        x = "Got {shape_repr(batch_a)} and {shape_repr(batch_b)}."
      ))
    }
  }

  # (C3)
  side <- if (left_side) shape_b[[rank_b - 1L]] else shape_b[[rank_b]]
  if (shape_a[[rank_a]] != side) {
    cli_abort(c(
      "{.arg a} and {.arg b} must agree on the axis the solve contracts over.",
      x = "Got {shape_repr(shape_a)} and {shape_repr(shape_b)}."
    ))
  }

  # (C4)
  list(AbstractArray(dtype = dtype(b), shape = b$shape))
}

infer_rng_bit_generator <- function(state, rng_algorithm, dtype, shape) {
  assert_array_dtype(state, "uint", naxes = 1L)
  if (dtype(state) != as_dtype("ui64")) {
    cli_abort(c(
      "{.arg state} must be {.val ui64}.",
      x = "Got {.val {as.character(dtype(state))}}."
    ))
  }

  assert_choice_param(rng_algorithm, "rng_algorithm", c("DEFAULT", "THREE_FRY", "PHILOX"))

  state_size <- shape(state)[[1L]]
  if (rng_algorithm == "THREE_FRY" && state_size != 2L) {
    cli_abort(c(
      "{.val THREE_FRY} requires an {.arg state} of length {.val {2L}}.",
      x = "Got {.val {state_size}}."
    ))
  }
  if (rng_algorithm == "PHILOX" && !(state_size %in% c(2L, 3L))) {
    cli_abort(c(
      "{.val PHILOX} requires an {.arg state} of length {.val {2L}} or {.val {3L}}.",
      x = "Got {.val {state_size}}."
    ))
  }

  out_dtype <- assert_dtype_param(dtype, "dtype")
  if (!any(vapply(c("int", "uint", "float"), dtype_in_category, logical(1L), dt = out_dtype))) {
    cli_abort(c(
      "{.arg dtype} must name {dtype_categories_repr(c('int', 'uint', 'float'))}.",
      x = "Got {.val {as.character(out_dtype)}}."
    ))
  }

  # (C1)
  list(
    state = AbstractArray(dtype = "ui64", shape = state$shape),
    values = AbstractArray(dtype = out_dtype, shape = Shape(assert_shapevec(shape)))
  )
}

infer_cholesky <- function(x, lower) {
  assert_array(x)
  # (I1), (C2), (C3) The same helper the other linalg rules use, so that one
  # mistake has one wording across them: float operands only (complex is not
  # supported yet), at least two axes with the last two square, and no
  # zero-sized axis. Axes before the last two are batch axes.
  assert_linalg_matrix(x, "x", square = TRUE, batched = TRUE)
  assert_flag_param(lower, "lower")

  # (C1)
  list(AbstractArray(dtype = dtype(x), shape = x$shape))
}

infer_qr <- function(x) {
  assert_array(x)
  assert_linalg_matrix(x, "x")
  s <- shape(x)
  m <- s[[1L]]
  n <- s[[2L]]
  k <- min(m, n)
  list(
    Q = AbstractArray(dtype = dtype(x), shape = Shape(c(m, k))),
    R = AbstractArray(dtype = dtype(x), shape = Shape(c(k, n)))
  )
}

infer_lu <- function(x) {
  assert_array(x)
  assert_linalg_matrix(x, "x")
  s <- shape(x)
  m <- s[[1L]]
  n <- s[[2L]]
  k <- min(m, n)
  index_dt <- default_int()
  list(
    LU = AbstractArray(dtype = dtype(x), shape = Shape(c(m, n))),
    pivots = AbstractArray(dtype = index_dt, shape = Shape(k)),
    permutation = AbstractArray(dtype = index_dt, shape = Shape(m))
  )
}

infer_svd <- function(x) {
  assert_array(x)
  assert_linalg_matrix(x, "x")
  s <- shape(x)
  m <- s[[1L]]
  n <- s[[2L]]
  k <- min(m, n)
  list(
    d = AbstractArray(dtype = dtype(x), shape = Shape(k)),
    u = AbstractArray(dtype = dtype(x), shape = Shape(c(m, k))),
    vt = AbstractArray(dtype = dtype(x), shape = Shape(c(k, n)))
  )
}

infer_eigh <- function(x) {
  assert_array(x)
  assert_linalg_matrix(x, "x", square = TRUE)
  n <- shape(x)[[1L]]
  # Names and order mirror `base::eigen()`: list(values, vectors).
  list(
    values = AbstractArray(dtype = dtype(x), shape = Shape(n)),
    vectors = AbstractArray(dtype = dtype(x), shape = Shape(c(n, n)))
  )
}

# ---------------------------------------------------------------------------
# Control flow
#
# Both branches of a conditional, and a loop's body, are already-traced anvl
# subgraphs, so their types are read straight off their input and output nodes.
# ---------------------------------------------------------------------------

infer_cond <- function(pred, true, false) {
  assert_array_dtype(pred, "bool", shape = integer())
  outs_true <- lapply(true$outputs, function(out) out$aval)
  outs_false <- lapply(false$outputs, function(out) out$aval)

  # Without this the mismatch is left to MLIR, which reports it as
  # `output_types(true_branch)[0]` over `tensor<f32>` -- StableHLO's vocabulary
  # and 0-based, for a program the caller wrote with `prim_if()`.
  if (length(outs_true) != length(outs_false)) {
    cli_abort(c(
      "{.arg true} and {.arg false} must return the same number of values.",
      x = "Got {length(outs_true)} and {length(outs_false)}."
    ))
  }
  bad <- which(
    !vapply(
      seq_along(outs_true),
      function(i) eq_type(outs_true[[i]], outs_false[[i]]),
      logical(1L)
    )
  )
  if (length(bad)) {
    described <- vapply(
      bad,
      function(i) {
        sprintf(
          "value %d is %s in `true` and %s in `false`",
          i,
          repr(outs_true[[i]]),
          repr(outs_false[[i]])
        )
      },
      character(1L)
    )
    cli_abort(
      c(
        "{.arg true} and {.arg false} must return the same type.",
        x = "{described}."
      ),
      call = NULL
    )
  }
  outs_true
}

infer_while <- function(..., cond, body) {
  outs <- list(...)
  outs_body <- lapply(body$outputs, function(out) out$aval)
  inputs_body <- lapply(body$inputs, function(inp) inp$aval)
  # `init` is a named list, so the loop state's names are the body's input
  # tree's child names. Read back rather than passed as a param: a param would
  # reach the lowering rules, which take the state positionally.
  state_names <- pjrt::tree_child_names(body$in_tree)

  # Names the state member that disagrees, since the usual cause is an R value
  # in `init` that materialized at its default: the loop is built before its
  # body runs, so the state cannot take a data type from it.
  labels <- if (length(state_names) == length(outs)) {
    state_names
  } else {
    sprintf("state %d", seq_along(outs))
  }
  mismatch <- function(a, b) {
    which(!vapply(seq_along(a), function(i) eq_type(a[[i]], b[[i]]), logical(1L)))
  }
  describe <- function(idx, a, b, verb) {
    vapply(
      idx,
      function(i) {
        sprintf(
          "`%s` %s %s and %s %s",
          labels[[i]],
          verb[[1L]],
          repr(a[[i]]),
          verb[[2L]],
          repr(b[[i]])
        )
      },
      character(1L)
    )
  }

  bad <- mismatch(outs, outs_body)
  if (length(bad)) {
    described <- describe(bad, outs, outs_body, c("enters as", "comes back as"))
    # The hint is about data types, so it is only shown when one actually
    # differs -- on a pure shape mismatch it points at the wrong fix.
    dtype_differs <- any(vapply(
      bad,
      function(i) dtype(outs[[i]]) != dtype(outs_body[[i]]),
      logical(1L)
    ))
    cli_abort(
      c(
        "{.arg init} and what {.arg body} returns must have the same type.",
        x = "{described}.",
        if (dtype_differs) {
          c(
            i = "An R value in {.arg init} materializes at its default data type; name the one the loop carries, e.g. {.code nv_scalar(0, dtype = \"f64\")} or {.fn nv_convert}."
          ) # nolint
        }
      ),
      call = NULL
    )
  }
  bad <- mismatch(inputs_body, outs_body)
  if (length(bad)) {
    described <- describe(bad, inputs_body, outs_body, c("is", "is returned as"))
    cli_abort(
      c(
        "{.arg body} must return the state it was given, unchanged in type.",
        x = "{described}."
      ),
      call = NULL
    )
  }

  # The body's outputs are what the loop carries, so those are the result.
  outs_body
}
