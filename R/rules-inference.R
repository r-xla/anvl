#' @include array.R
#' @include default-dtypes.R
NULL

# Type inference for every primitive: given the input `AbstractArray`s and the
# static params, return the output `AbstractArray`s (a list, named when the
# primitive has named outputs), or refuse the call.
#
# `graph_desc_add()` calls a rule as `do.call(infer_fn, c(avals_in, params))`,
# so its formals are exactly the primitive's arguments, used or not, and its
# messages name arguments as the `prim_*()` function spells them. Every check
# that can be decided from the avals and params lives here rather than in the
# `prim_*()` body, so each mistake has one wording. A message names the
# offending argument and reports what it was given via `value_repr()`.
#
# Constraint numbers such as (C1) and (I2) refer to the StableHLO spec.

# ---------------------------------------------------------------------------
# Shared checking helpers
# ---------------------------------------------------------------------------

repr_max_entries <- 8L
repr_max_chars <- 30L

# A value the caller passed, as an error message reports it. Unlike
# `format_param()`, this copes with anything: long vectors and strings are
# truncated, matrices and arrays read as the call that builds them, and other
# objects are reported by class.
value_repr <- function(x) {
  if (is.null(x) || identical(x, list())) {
    return(format_param(x))
  }
  # Under `jit()` an operand is a `GraphBox`; report its array type instead.
  if (is_arrayish(x, convert_ok = FALSE)) {
    return(repr(AbstractArray(dtype = dtype(x), shape = shape(x))))
  }
  if (!is.atomic(x) || is.object(x)) {
    if (is.vector(x)) {
      return(cli::format_inline("{.cls {class(x)[1L]}} of length {length(x)}"))
    }
    return(cli::format_inline("{.cls {class(x)[1L]}}"))
  }
  shape <- dim(x)
  x <- as.vector(x)
  n <- length(x)
  entries <- value_entries_repr(x[seq_len(min(n, repr_max_entries))])
  if (n > 1L) {
    entries <- paste0("c(", entries, if (n > repr_max_entries) ", ...", ")")
  }
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

# The entries of a short atomic vector, comma-separated, with long strings cut
# and missing values spelled `NA`.
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

# A whole-number param, returned as an integer vector. `NULL` (how `c()` spells
# no axes) reads as `integer()`. Rules run this before indexing or comparing
# with a param, so that an `NA` never reaches an `if ()`.
assert_int_param <- function(x, arg = rlang::as_label(substitute(x)), len = NULL, min_len = NULL) {
  force(arg)
  x <- x %||% integer()
  # A param fixed at one entry is spoken of in the singular.
  one <- identical(len, 1L)
  whole <- if (one) "be a whole number" else "contain whole numbers"
  problem <- if (!is.numeric(x) || is.object(x)) {
    paste0("be a whole number", if (!one) " vector")
  } else if (anyNA(x)) {
    if (one) "not be a missing value" else "not contain missing values"
  } else if (any(x != trunc(x))) {
    whole
  } else if (any(abs(x) > .Machine$integer.max)) {
    paste(whole, "in the integer range")
  } else if (!is.null(len) && length(x) != len) {
    "have {len} entr{cli::qty(len)}{?y/ies}"
  } else if (!is.null(min_len) && length(x) < min_len) {
    "have at least {min_len} entr{cli::qty(min_len)}{?y/ies}"
  }
  if (!is.null(problem)) {
    cli_abort(c(
      paste0("{.arg {arg}} must ", problem, "."),
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(as.integer(x))
}

# A data type the caller named, optionally restricted to some categories.
assert_dtype_param <- function(x, arg = rlang::as_label(substitute(x)), categories = NULL) {
  out <- tryCatch(as_dtype(x), error = function(e) NULL)
  if (is.null(out)) {
    cli_abort(c(
      "{.arg {arg}} must name a data type.",
      x = "Got {value_repr(x)}.",
      i = "See {.fn tengen::as_dtype} for the data types anvl knows."
    ))
  }
  if (!is.null(categories) && !dtype_in_categories(out, categories)) {
    cli_abort(c(
      "{.arg {arg}} must name {dtype_categories_repr(categories)}.",
      x = "Got {.val {as.character(out)}}."
    ))
  }
  out
}

assert_flag_param <- function(x, arg = rlang::as_label(substitute(x))) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    cli_abort(c(
      "{.arg {arg}} must be {.val {TRUE}} or {.val {FALSE}}.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(x)
}

assert_choice_param <- function(x, choices, arg = rlang::as_label(substitute(x))) {
  if (!rlang::is_string(x) || !(x %in% choices)) {
    cli_abort(c(
      "{.arg {arg}} must be one of {.or {.val {choices}}}.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(x)
}

# Sizes that become a shape, so they must also be non-negative.
assert_size_param <- function(x, arg = rlang::as_label(substitute(x)), len = NULL) {
  x <- assert_int_param(x, arg, len = len)
  if (any(x < 0L)) {
    cli_abort(c(
      "{.arg {arg}} must not be negative.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(x)
}

# A result shape computed in double from the caller's `parts` (params that fit
# an integer one by one can still overflow in sum). Returns it as integer.
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

# "a float data type", "an integer or unsigned integer data type".
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

# Does `dt` belong to any of the categories "float", "int", "uint", "bool"?
dtype_in_categories <- function(dt, categories) {
  checks <- list(
    float = is_dtype_float,
    int = is_dtype_int,
    uint = is_dtype_uint,
    bool = is_dtype_bool
  )
  any(vapply(checks[categories], function(check) check(dt), logical(1L)))
}

assert_array <- function(x, arg = rlang::as_label(substitute(x))) {
  if (!inherits(x, "AbstractArray")) {
    cli_abort(c(
      "{.arg {arg}} must be an array.",
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(NULL)
}

# Check several operands at once. Each is named in messages by the name it
# arrived under, as `.arg[[i]]` when the primitive collects them in one
# argument (`prim_sort()`'s `xs`), and otherwise as `..i`.
assert_arrays <- function(..., .arg = NULL) {
  args <- list(...)
  arg_names <- if (is.null(.arg)) {
    rlang::names2(args)
  } else {
    sprintf("%s[[%i]]", .arg, seq_along(args))
  }
  unnamed <- !nzchar(arg_names)
  arg_names[unnamed] <- paste0("..", which(unnamed))
  for (i in seq_along(args)) {
    assert_array(args[[i]], arg = arg_names[[i]])
  }
  invisible(NULL)
}

# Check an array's data type against the categories in `...`, and optionally
# its shape or number of axes.
assert_array_dtype <- function(
  x,
  ...,
  shape = NULL,
  naxes = NULL,
  arg = rlang::as_label(substitute(x))
) {
  assert_array(x, arg = arg)
  categories <- c(...)

  if (length(categories) && !dtype_in_categories(dtype(x), categories)) {
    cli_abort(c(
      "{.arg {arg}} must have {dtype_categories_repr(categories)}.",
      x = "Got {.val {as.character(dtype(x))}}."
    ))
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
  arg_x = rlang::as_label(substitute(x)),
  arg_y = rlang::as_label(substitute(y))
) {
  assert_array(x, arg = arg_x)
  assert_array(y, arg = arg_y)
  if (!eq_type(x, y)) {
    cli_abort(c(
      "{.arg {arg_x}} and {.arg {arg_y}} must have the same array type.",
      x = "Got {repr(x)} and {repr(y)}."
    ))
  }
  invisible(NULL)
}

assert_same_dtype <- function(
  x,
  y,
  arg_x = rlang::as_label(substitute(x)),
  arg_y = rlang::as_label(substitute(y))
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

assert_axes_in_range <- function(axes, n_axes, arg = rlang::as_label(substitute(axes))) {
  if (!length(axes)) {
    return(invisible(NULL))
  }
  if (n_axes == 0L) {
    cli_abort(c(
      "{.arg {arg}} cannot be used, there is no axis to select.",
      x = "Got {value_repr(axes)}."
    ))
  }
  if (any(axes < 1L) || any(axes > n_axes)) {
    cli_abort(c(
      "{.arg {arg}} must contain axes between {.val {1L}} and {.val {n_axes}}.",
      x = "Got {value_repr(axes)}."
    ))
  }
  invisible(NULL)
}

assert_axes_unique <- function(axes, arg = rlang::as_label(substitute(axes))) {
  if (anyDuplicated(axes)) {
    cli_abort(c(
      "{.arg {arg}} must contain unique axes.",
      x = "Got {value_repr(axes)}."
    ))
  }
  invisible(NULL)
}

assert_axes_sorted <- function(axes, arg = rlang::as_label(substitute(axes))) {
  if (is.unsorted(axes)) {
    cli_abort(c(
      "{.arg {arg}} must be sorted in ascending order.",
      x = "Got {value_repr(axes)}."
    ))
  }
  invisible(NULL)
}

# A reduction's `axes`: whole numbers naming distinct axes of `x`.
assert_reduction_axes <- function(x, axes) {
  axes <- assert_int_param(axes)
  assert_axes_in_range(axes, length(shape(x)))
  assert_axes_unique(axes)
  axes
}

# A param with one entry per axis, e.g. `slice_sizes` or `window_strides`.
assert_entry_per_axis <- function(x, n, arg = rlang::as_label(substitute(x)), axes = "axis of {.arg x}") {
  if (length(x) != n) {
    cli_abort(c(
      paste0("{.arg {arg}} must have one entry per ", axes, " ({n})."),
      x = "Got {value_repr(x)}."
    ))
  }
  invisible(NULL)
}

# A layout: several params that between them name every axis of one array
# exactly once (`prim_convolution()`'s three).
assert_axis_layout <- function(parts, n_axes, what) {
  parts <- Map(assert_int_param, parts, names(parts))
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
  if (length(axes) != n_axes) {
    cli_abort(c(
      "The axes of {what} must each be named exactly once.",
      x = "{.arg {names(parts)}} name {cli::qty(length(axes))}{length(axes)} ax{?is/es} between them, but {what} has {n_axes}.", # nolint
      i = "Got {params_repr(parts)}."
    ))
  }
  for (nm in names(parts)) {
    assert_axes_in_range(parts[[nm]], n_axes, nm)
  }
  invisible(NULL)
}

# A sub-graph traced against two scalars of `x`'s data type (`prim_reduce()`'s
# `reducer`, `prim_scatter()`'s `update_fn`) must return one such scalar.
assert_scalar_fn_output <- function(graph, x, arg = rlang::as_label(substitute(graph))) {
  outputs <- lapply(graph$outputs, function(out) out$aval)
  if (length(outputs) != 1L) {
    cli_abort(c(
      "{.arg {arg}} must return exactly one value.",
      x = "Got {length(outputs)} outputs."
    ))
  }
  out <- outputs[[1L]]
  if (dtype(out) != dtype(x)) {
    cli_abort(c(
      "{.arg {arg}} must return a value with the same data type as {.arg x}.",
      x = "{.arg x} is {.val {as.character(dtype(x))}}, but {.arg {arg}} returns {.val {as.character(dtype(out))}}." # nolint
    ))
  }
  if (length(shape(out))) {
    cli_abort(c(
      "{.arg {arg}} must return a scalar.",
      x = "Got shape {shape_repr(shape(out))}."
    ))
  }
  invisible(NULL)
}

# Unlike `setequal()`, this respects multiplicity and length.
test_permutation <- function(x, expected) {
  length(x) == length(expected) && setequal(x, expected) && !anyDuplicated(x)
}

# Index of the first of `values` that differs from the first one, or `NULL`.
first_mismatch <- function(values) {
  bad <- which(!vapply(values, identical, logical(1L), values[[1L]]))
  if (length(bad)) bad[[1L]]
}

dtype_names <- function(xs) {
  vapply(xs, function(x) as.character(dtype(x)), character(1L))
}

# ---------------------------------------------------------------------------
# Element-wise rules
# ---------------------------------------------------------------------------

# `arg_lhs` / `arg_rhs` are the primitive's names for its operands, passed by
# `make_binary_op()`: `x` / `y` for `prim_pow()`, `x` / `shift` for the shifts.
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

# The bit shifts, which unlike `and` / `or` / `xor` exclude `bool`.
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

# `prim_abs()` and `prim_sign()`, which have no use for unsigned input.
infer_signed_uni <- function(x) {
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

infer_round <- function(x, method) {
  assert_choice_param(method, c("nearest_even", "afz"))
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
  list(AbstractArray(dtype = assert_dtype_param(dtype), shape = x$shape))
}

infer_bitcast_convert <- function(x, dtype) {
  assert_array(x)
  in_dtype <- dtype(x)
  out_dtype <- assert_dtype_param(dtype)

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

  # (C1), (C2) A wider target packs a leading axis of `out_width / in_width`
  # elements into one; a narrower one unpacks into a new leading axis.
  # (StableHLO uses the last axis; anvl the first, to match column-major order.)
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
    result_shape <- c(in_width %/% out_width, in_shape)
  }

  list(AbstractArray(dtype = out_dtype, shape = Shape(result_shape)))
}

infer_broadcast_in_axes <- function(x, shape, broadcast_axes) {
  assert_array(x)
  shape <- assert_shapevec(shape)
  in_shape <- shape(x)
  broadcast_axes <- assert_int_param(broadcast_axes)

  # (C2)
  if (length(broadcast_axes) != length(in_shape)) {
    cli_abort(c(
      "{.arg broadcast_axes} must have one entry per axis of {.arg x}.",
      x = "Got {value_repr(broadcast_axes)} for an {.arg x} with {cli::qty(length(in_shape))}{length(in_shape)} ax{?is/es}." # nolint
    ))
  }

  # (C3), (C4)
  assert_axes_in_range(broadcast_axes, length(shape))
  assert_axes_unique(broadcast_axes)

  # (C5)
  for (d in seq_along(broadcast_axes)) {
    from <- in_shape[[d]]
    to <- shape[[broadcast_axes[[d]]]]
    if (from != to && from != 1L) {
      cli_abort(c(
        "Axis {d} of {.arg x} must be {.or {.val {unique(c(to, 1L))}}} to broadcast to axis {broadcast_axes[[d]]} of the result.", # nolint
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
  perm <- assert_int_param(perm)

  # (C2)
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
  result_shape <- assert_size_param(shape)

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
  dtypes <- dtype_names(xs)
  bad <- first_mismatch(dtypes)
  if (!is.null(bad)) {
    bad_arg <- paste0("..", bad)
    cli_abort(c(
      "Every input must have the same data type.",
      x = "{.arg ..1} is {.val {dtypes[[1L]]}}, {.arg {bad_arg}} is {.val {dtypes[[bad]]}}."
    ))
  }

  # (C4)
  n_axes <- length(shapes[[1L]])
  axis <- assert_int_param(axis, len = 1L)
  assert_axes_in_range(axis, n_axes)

  # (C2) Ranks first: `s[-axis]` below would silently drop nothing from a
  # shape with fewer than `axis` axes.
  input_n_axes <- lengths(shapes)
  bad <- first_mismatch(input_n_axes)
  if (!is.null(bad)) {
    bad_arg <- paste0("..", bad)
    cli_abort(c(
      "Every input must have the same number of axes.",
      x = "{.arg ..1} has {cli::qty(n_axes)}{n_axes} ax{?is/es} {shape_repr(shapes[[1L]])}, {.arg {bad_arg}} has {input_n_axes[[bad]]} {shape_repr(shapes[[bad]])}." # nolint
    ))
  }
  bad <- first_mismatch(lapply(shapes, function(s) s[-axis]))
  if (!is.null(bad)) {
    bad_arg <- paste0("..", bad)
    cli_abort(c(
      "Every input must have the same shape except along {.arg axis} ({axis}).",
      x = "{.arg ..1} has shape {shape_repr(shapes[[1L]])}, {.arg {bad_arg}} has shape {shape_repr(shapes[[bad]])}." # nolint
    ))
  }

  # (C6)
  result_shape <- shapes[[1L]]
  result_shape[axis] <- sum(vapply(shapes, function(s) s[[axis]], integer(1L)))

  list(AbstractArray(dtype = dtype(xs[[1L]]), shape = Shape(result_shape)))
}

infer_reverse <- function(x, axes) {
  assert_array(x)
  axes <- assert_int_param(axes)

  # (C2), (C3) An empty `axes` is allowed.
  assert_axes_unique(axes)
  assert_axes_in_range(axes, length(shape(x)))

  list(AbstractArray(dtype = dtype(x), shape = x$shape))
}

# `end_indices` is the last index kept.
infer_static_slice <- function(x, start_indices, end_indices, strides) {
  assert_array(x)
  start <- assert_int_param(start_indices)
  end <- assert_int_param(end_indices)
  stride <- assert_int_param(strides)
  in_shape <- shape(x)
  n_axes <- length(in_shape)

  # (C2)
  given <- list(start_indices = start, end_indices = end, strides = stride)
  if (any(lengths(given) != n_axes)) {
    cli_abort(c(
      "{.arg start_indices}, {.arg end_indices} and {.arg strides} must have one entry per axis of {.arg x} ({n_axes}).", # nolint
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
  # Before the next check, so that `end + 1L` cannot overflow.
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

  # (C4)
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
  low <- assert_int_param(edge_padding_low)
  high <- assert_int_param(edge_padding_high)
  interior <- assert_int_param(interior_padding)
  given <- list(edge_padding_low = low, edge_padding_high = high, interior_padding = interior)

  # (C2)
  for (nm in names(given)) {
    assert_entry_per_axis(given[[nm]], length(in_shape), nm)
  }

  # (C3)
  if (any(interior < 0L)) {
    cli_abort(c(
      "{.arg interior_padding} must be non-negative.",
      x = "Got {value_repr(interior)}."
    ))
  }

  # (C4)
  result_shape <- in_shape + low + pmax(in_shape - 1L, 0L) * interior + high
  if (any(result_shape < 0L)) {
    bad <- which(result_shape < 0L)
    cli_abort(c(
      "Negative padding must not remove more than an axis holds.",
      x = "{.arg x} has shape {shape_repr(in_shape)}; {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)} would end up at {value_repr(result_shape[bad])}.", # nolint
      i = "Got {params_repr(given)}."
    ))
  }

  list(AbstractArray(dtype = dtype(x), shape = Shape(result_shape)))
}

infer_select <- function(test, yes, no) {
  # (C2)
  assert_same_type(yes, no)
  assert_array_dtype(test, "bool")

  # (C1) `test` is either element-wise or a single scalar switch.
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
  assert_same_dtype(x, max)
  assert_same_dtype(min, x)

  # (C1)
  bounds <- list(min = min, max = max)
  for (nm in names(bounds)) {
    bound_shape <- shape(bounds[[nm]])
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
  axis <- assert_int_param(axis, len = 1L)
  start <- assert_int_param(start, len = 1L)

  # (C1)
  assert_axes_in_range(axis, length(shape))
  dtype <- assert_dtype_param(dtype, categories = c("int", "uint", "float"))

  list(IotaArray(shape = shape, dtype = dtype, axis = axis, start = start))
}

# `value` is the R scalar to fill with, not an operand.
infer_fill <- function(value, shape, dtype) {
  if (inherits(value, "AbstractArray") || length(value) != 1L) {
    cli_abort(c(
      "{.arg value} must be a scalar.",
      x = "Got {value_repr(value)}."
    ))
  }
  list(AbstractArray(
    dtype = assert_dtype_param(dtype),
    shape = Shape(assert_shapevec(shape))
  ))
}

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
  assert_choice_param(precision, c("default", "high", "highest"))

  shape_lhs <- shape(lhs)
  shape_rhs <- shape(rhs)

  axis_pairs <- list(contracting_axes = contracting_axes, batching_axes = batching_axes)
  for (nm in names(axis_pairs)) {
    val <- axis_pairs[[nm]]
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
  assert_axes_in_range(lhs_batching, length(shape_lhs), "batching_axes[[1]]")
  assert_axes_in_range(lhs_contracting, length(shape_lhs), "contracting_axes[[1]]")
  assert_axes_in_range(rhs_batching, length(shape_rhs), "batching_axes[[2]]")
  assert_axes_in_range(rhs_contracting, length(shape_rhs), "contracting_axes[[2]]")

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

  # (C12)
  result_shape <- c(
    size_batch_lhs,
    without(shape_lhs, c(lhs_contracting, lhs_batching)),
    without(shape_rhs, c(rhs_contracting, rhs_batching))
  )

  list(AbstractArray(dtype = dtype(lhs), shape = Shape(result_shape)))
}

# Every start index is a scalar of the same integer type, one per axis.
assert_start_indices <- function(start_indices, n_axes) {
  # (C2) / (C4)
  if (length(start_indices) != n_axes) {
    cli_abort(c(
      "{.arg ...} must hold one start index per axis of {.arg x} ({n_axes}).",
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
  dtypes <- dtype_names(start_indices)
  bad <- first_mismatch(dtypes)
  if (!is.null(bad)) {
    bad_arg <- sprintf("..%d", bad)
    cli_abort(c(
      "Every start index must have the same data type.",
      x = "{.arg ..1} is {.val {dtypes[[1L]]}} and {.arg {bad_arg}} is {.val {dtypes[[bad]]}}."
    ))
  }
  invisible(NULL)
}

infer_dynamic_slice <- function(x, ..., slice_sizes) {
  assert_array(x)
  in_shape <- shape(x)
  sizes <- assert_size_param(slice_sizes)

  assert_start_indices(list(...), length(in_shape))

  # (C2)
  assert_entry_per_axis(sizes, length(in_shape), "slice_sizes")

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
  in_shape <- shape(x)
  n_axes <- length(in_shape)

  assert_start_indices(list(...), n_axes)

  # (C2)
  assert_same_dtype(x, update)

  # (C3)
  if (length(shape(update)) != n_axes) {
    cli_abort(c(
      "{.arg update} must have as many axes as {.arg x} ({n_axes}).",
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
  k <- assert_int_param(k, len = 1L)
  assert_flag_param(indices)

  in_shape <- shape(x)
  n_axes <- length(in_shape)

  if (n_axes < 1L) {
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
  last <- in_shape[[n_axes]]
  if (k > last) {
    cli_abort(c(
      "{.arg k} must not exceed the size of the last axis of {.arg x} ({last}).",
      x = "Got {.val {k}}."
    ))
  }

  result_shape <- in_shape
  result_shape[[n_axes]] <- k

  values <- AbstractArray(dtype = dtype(x), shape = Shape(result_shape))
  if (!indices) {
    return(list(values = values))
  }
  list(
    values = values,
    indices = AbstractArray(dtype = default_int(), shape = Shape(result_shape))
  )
}

# The shape a reduction leaves behind: `axes` dropped, or kept at size 1.
reduced_shape <- function(x, axes, drop) {
  new_shape <- shape(x)
  if (drop) {
    return(without(new_shape, axes))
  }
  new_shape[axes] <- 1L
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

  assert_flag_param(drop)

  # (C3)
  assert_same_dtype(x, init)

  # (C4), (C5)
  axes <- assert_reduction_axes(x, axes)

  # (C6)
  assert_scalar_fn_output(reducer, x)

  # (C7)
  list(AbstractArray(dtype = dtype(x), shape = Shape(reduced_shape(x, axes, drop))))
}

# `prim_sum()`, `prim_max()` and friends, whose reducer is fixed.
infer_reduce_simple <- function(x, axes, drop) {
  assert_array(x)
  assert_flag_param(drop)
  axes <- assert_reduction_axes(x, axes)
  list(AbstractArray(dtype = dtype(x), shape = Shape(reduced_shape(x, axes, drop))))
}

infer_reduce_boolean <- function(x, axes, drop) {
  assert_array(x)
  assert_flag_param(drop)
  axes <- assert_reduction_axes(x, axes)
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
  list(AbstractArray(dtype = "bool", shape = Shape(reduced_shape(x, axes, drop))))
}

# The axis a scan (`prim_cumsum()` and friends) runs along.
assert_scan_axis <- function(x, axis) {
  n_axes <- length(shape(x))
  if (n_axes == 0L) {
    cli_abort(c(
      "{.arg x} must have at least one axis to accumulate along.",
      x = "Got a scalar."
    ))
  }
  axis <- assert_int_param(axis, len = 1L)
  assert_axes_in_range(axis, n_axes)
}

infer_cum <- function(x, axis) {
  assert_array(x)
  assert_scan_axis(x, axis)
  list(AbstractArray(dtype = dtype(x), shape = x$shape))
}

# `prim_cummax()` / `prim_cummin()`: the running extremum and where it sits.
infer_cum_extreme <- function(x, axis) {
  assert_array(x)
  assert_scan_axis(x, axis)
  list(
    values = AbstractArray(dtype = dtype(x), shape = x$shape),
    indices = AbstractArray(dtype = default_int(), shape = x$shape)
  )
}

# `prim_which_max()` / `prim_which_min()`.
infer_arg_extreme <- function(x, axis, drop) {
  assert_array(x)
  assert_flag_param(drop)
  axis <- assert_int_param(axis, len = 1L)
  shp <- shape(x)
  assert_axes_in_range(axis, length(shp))
  # The index of an extremum of nothing is undefined; the lowering would
  # silently return its init value instead.
  if (shp[axis] == 0L) {
    cli_abort(c(
      "{.arg x} must have elements along the axis this reads.",
      x = "{.arg x} has shape {shape_repr(shp)}; axis {axis} has size 0."
    ))
  }
  list(AbstractArray(dtype = default_int(), shape = Shape(reduced_shape(x, axis, drop))))
}

# The operands arrive separately, but `prim_sort()` takes them as `xs`.
infer_sort <- function(..., axis, decreasing, stable) {
  xs <- list(...)
  if (!length(xs)) {
    cli_abort(c(
      "{.arg xs} must be a non-empty list of arrayish values.",
      x = "Got nothing to sort."
    ))
  }
  assert_flag_param(decreasing)
  assert_flag_param(stable)
  assert_arrays(..., .arg = "xs")

  # (C1), (C2)
  shapes <- lapply(xs, shape)
  bad <- first_mismatch(shapes)
  if (!is.null(bad)) {
    cli_abort(c(
      "Every element of {.arg xs} must have the same shape.",
      x = "{.arg xs[[1]]} has shape {shape_repr(shapes[[1L]])}, {.arg xs[[{bad}]]} has shape {shape_repr(shapes[[bad]])}." # nolint
    ))
  }

  axis <- assert_int_param(axis, len = 1L)
  assert_axes_in_range(axis, length(shapes[[1L]]))
  lapply(xs, function(x) AbstractArray(dtype = dtype(x), shape = x$shape))
}

# ---------------------------------------------------------------------------
# Gather and scatter
#
# `index_vector_axis` may be `n_axes + 1`, meaning the index vectors are
# implicit: each entry of the indices is a single index.
# ---------------------------------------------------------------------------

assert_index_vector_axis <- function(index_vector_axis, idx_n_axes) {
  if (index_vector_axis < 1L || index_vector_axis > idx_n_axes + 1L) {
    cli_abort(c(
      "{.arg index_vector_axis} must be between {.val {1L}} and {.val {idx_n_axes + 1L}}.",
      x = "Got {.val {index_vector_axis}}."
    ))
  }
  invisible(NULL)
}

# The number of coordinates in each index vector.
index_vector_size <- function(idx_shape, index_vector_axis) {
  if (index_vector_axis <= length(idx_shape)) idx_shape[[index_vector_axis]] else 1L
}

# The batching axes of the indices (`idx_arg`) pair up with those of `x`.
assert_index_batching_axes <- function(
  x_shape,
  idx_shape,
  x_batching_axes,
  idx_batching_axes,
  index_vector_axis,
  idx_arg
) {
  batching_arg <- paste0(idx_arg, "_batching_axes")
  assert_axes_unique(idx_batching_axes, batching_arg)
  assert_axes_in_range(idx_batching_axes, length(idx_shape), batching_arg)

  if (index_vector_axis %in% idx_batching_axes) {
    cli_abort(c(
      "{.arg index_vector_axis} must not be one of {.arg {batching_arg}}.",
      x = "{.arg index_vector_axis} is {.val {index_vector_axis}} and {.arg {batching_arg}} is {value_repr(idx_batching_axes)}." # nolint
    ))
  }
  if (length(x_batching_axes) != length(idx_batching_axes)) {
    cli_abort(c(
      "{.arg x_batching_axes} and {.arg {batching_arg}} must have the same length.",
      x = "Got {value_repr(x_batching_axes)} and {value_repr(idx_batching_axes)}."
    ))
  }
  batch_x <- x_shape[x_batching_axes]
  batch_idx <- idx_shape[idx_batching_axes]
  if (!identical(batch_x, batch_idx)) {
    cli_abort(c(
      "The batching axes of {.arg x} and {.arg {idx_arg}} must have the same sizes.",
      x = "Got {shape_repr(batch_x)} and {shape_repr(batch_idx)}."
    ))
  }
  invisible(NULL)
}

# A gather that drops an axis from the result must slice it at size one.
assert_unit_slice_sizes <- function(sizes, axes, arg = rlang::as_label(substitute(axes))) {
  bad <- axes[sizes[axes] > 1L]
  if (length(bad)) {
    cli_abort(c(
      "{.arg slice_sizes} must be at most {.val {1L}} at {.arg {arg}}.",
      x = "Got {value_repr(sizes[bad])} at {cli::qty(length(bad))}ax{?is/es} {value_repr(bad)}."
    ))
  }
  invisible(NULL)
}

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
  x_n_axes <- length(x_shape)
  idx_shape <- shape(start_indices)
  sizes <- assert_size_param(slice_sizes)
  index_vector_axis <- assert_int_param(index_vector_axis, len = 1L)
  assert_flag_param(indices_are_sorted)
  assert_flag_param(unique_indices)

  offset_axes <- assert_int_param(offset_axes)
  collapsed_slice_axes <- assert_int_param(collapsed_slice_axes)
  x_batching_axes <- assert_int_param(x_batching_axes)
  start_indices_batching_axes <- assert_int_param(start_indices_batching_axes)
  start_index_map <- assert_int_param(start_index_map)

  # (C20) first, since the number of result axes below is computed from `slice_sizes`.
  assert_entry_per_axis(sizes, x_n_axes, "slice_sizes")

  # (C1)
  expected_n_axes <- length(offset_axes) +
    length(collapsed_slice_axes) +
    length(x_batching_axes)
  if (x_n_axes != expected_n_axes) {
    cli_abort(c(
      "{.arg x} must have one axis per entry of {.arg offset_axes}, {.arg collapsed_slice_axes} and {.arg x_batching_axes}.", # nolint
      x = "{.arg x} has {cli::qty(x_n_axes)}{x_n_axes} ax{?is/es}, but those name {expected_n_axes} ({length(offset_axes)} + {length(collapsed_slice_axes)} + {length(x_batching_axes)})." # nolint
    ))
  }

  # (C2)
  assert_index_vector_axis(index_vector_axis, length(idx_shape))

  # (C3)
  map_size <- index_vector_size(idx_shape, index_vector_axis)
  if (length(start_index_map) != map_size) {
    cli_abort(c(
      "{.arg start_index_map} must have one entry per index coordinate ({map_size}).",
      x = "Got {value_repr(start_index_map)}."
    ))
  }

  # (C4)
  assert_axes_unique(offset_axes)
  assert_axes_sorted(offset_axes)

  batch_sizes <- without(idx_shape, index_vector_axis)
  offset_sizes <- without(sizes, c(collapsed_slice_axes, x_batching_axes))
  result_n_axes <- length(batch_sizes) + length(offset_sizes)

  # (C5)
  assert_axes_in_range(offset_axes, result_n_axes)

  # (C6)
  assert_axes_unique(
    c(collapsed_slice_axes, x_batching_axes),
    "collapsed_slice_axes and x_batching_axes"
  )

  # (C7) - (C9)
  assert_axes_sorted(collapsed_slice_axes)
  assert_axes_in_range(collapsed_slice_axes, x_n_axes)
  assert_unit_slice_sizes(sizes, collapsed_slice_axes)

  # (C10) - (C12)
  assert_axes_sorted(x_batching_axes)
  assert_axes_in_range(x_batching_axes, x_n_axes)
  assert_unit_slice_sizes(sizes, x_batching_axes)

  # (C13) - (C17)
  assert_index_batching_axes(
    x_shape,
    idx_shape,
    x_batching_axes,
    start_indices_batching_axes,
    index_vector_axis,
    "start_indices"
  )

  # (C18), (C19)
  assert_axes_unique(
    c(start_index_map, x_batching_axes),
    "start_index_map and x_batching_axes"
  )
  assert_axes_in_range(start_index_map, x_n_axes)

  # (C21)
  if (any(sizes > x_shape)) {
    cli_abort(c(
      "{.arg slice_sizes} must be between {.val {0L}} and the shape of {.arg x} {shape_repr(x_shape)}.",
      x = "Got {value_repr(sizes)}."
    ))
  }

  # (C22) The batch axes fill the result axes that `offset_axes` leaves free.
  result_shape <- integer(result_n_axes)
  result_shape[setdiff(seq_len(result_n_axes), offset_axes)] <- batch_sizes
  result_shape[offset_axes] <- offset_sizes

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
  x_n_axes <- length(x_shape)
  idx_shape <- shape(scatter_indices)
  update_shape <- shape(update)
  update_n_axes <- length(update_shape)

  index_vector_axis <- assert_int_param(index_vector_axis, len = 1L)
  assert_flag_param(indices_are_sorted)
  assert_flag_param(unique_indices)

  update_window_axes <- assert_int_param(update_window_axes)
  inserted_window_axes <- assert_int_param(inserted_window_axes)
  x_batching_axes <- assert_int_param(x_batching_axes)
  scatter_indices_batching_axes <- assert_int_param(scatter_indices_batching_axes)
  scatter_axes_to_x_axes <- assert_int_param(scatter_axes_to_x_axes)

  # (C7) before (C2), whose count a repeated entry would throw off.
  assert_axes_unique(update_window_axes)

  # (C2)
  expected_n_axes <- length(update_window_axes) +
    length(inserted_window_axes) +
    length(x_batching_axes)
  if (x_n_axes != expected_n_axes) {
    cli_abort(c(
      "{.arg x} must have one axis per entry of {.arg update_window_axes}, {.arg inserted_window_axes} and {.arg x_batching_axes}.", # nolint
      x = "{.arg x} has {cli::qty(x_n_axes)}{x_n_axes} ax{?is/es}, but those name {expected_n_axes} ({length(update_window_axes)} + {length(inserted_window_axes)} + {length(x_batching_axes)})." # nolint
    ))
  }

  # (C6)
  assert_same_dtype(x, update)

  # (C22)
  assert_index_vector_axis(index_vector_axis, length(idx_shape))

  # (C4) `update` has one axis per scatter axis plus one per window axis.
  scatter_sizes <- without(idx_shape, index_vector_axis)
  expected_update_n_axes <- length(scatter_sizes) + length(update_window_axes)
  if (update_n_axes != expected_update_n_axes) {
    cli_abort(c(
      "{.arg update} must have {cli::qty(expected_update_n_axes)}{expected_update_n_axes} ax{?is/es}.",
      x = "Got {shape_repr(update_shape)}."
    ))
  }

  # (C8)
  assert_axes_sorted(update_window_axes)
  assert_axes_in_range(update_window_axes, update_n_axes)

  # (C9) - (C11)
  assert_axes_unique(
    c(inserted_window_axes, x_batching_axes),
    "inserted_window_axes and x_batching_axes"
  )
  assert_axes_sorted(inserted_window_axes)
  assert_axes_in_range(inserted_window_axes, x_n_axes)

  # (C12), (C13)
  assert_axes_sorted(x_batching_axes)
  assert_axes_in_range(x_batching_axes, x_n_axes)

  # (C14) - (C18)
  assert_index_batching_axes(
    x_shape,
    idx_shape,
    x_batching_axes,
    scatter_indices_batching_axes,
    index_vector_axis,
    "scatter_indices"
  )

  # (C19)
  map_size <- index_vector_size(idx_shape, index_vector_axis)
  if (length(scatter_axes_to_x_axes) != map_size) {
    cli_abort(c(
      "{.arg scatter_axes_to_x_axes} must have one entry per index coordinate ({map_size}).",
      x = "Got {value_repr(scatter_axes_to_x_axes)}."
    ))
  }

  # (C20), (C21)
  assert_axes_unique(
    c(scatter_axes_to_x_axes, x_batching_axes),
    "scatter_axes_to_x_axes and x_batching_axes"
  )
  assert_axes_in_range(scatter_axes_to_x_axes, x_n_axes)

  window_sizes <- without(x_shape, c(inserted_window_axes, x_batching_axes))
  actual_window <- update_shape[update_window_axes]
  if (any(actual_window > window_sizes)) {
    cli_abort(c(
      "{.arg update} must not be larger than {.arg x} along its window axes.",
      x = "Got {value_repr(actual_window)}, at most {value_repr(window_sizes)} is allowed."
    ))
  }

  actual_scatter <- update_shape[setdiff(seq_len(update_n_axes), update_window_axes)]
  if (!identical(actual_scatter, scatter_sizes)) {
    cli_abort(c(
      "The scatter axes of {.arg update} must match the shape of {.arg scatter_indices}.",
      x = "Got {value_repr(actual_scatter)}, expected {value_repr(scatter_sizes)}."
    ))
  }

  # (C23)
  assert_scalar_fn_output(update_fn, x)

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
  n_axes <- length(x_shape)

  # (C1)
  if (n_axes != length(kernel_shape)) {
    cli_abort(c(
      "{.arg x} and {.arg kernel} must have the same number of axes.",
      x = "Got {shape_repr(x_shape)} and {shape_repr(kernel_shape)}."
    ))
  }
  if (n_axes < 2L) {
    cli_abort(c(
      "{.arg x} and {.arg kernel} must have at least two axes.",
      x = "Got {shape_repr(x_shape)} and {shape_repr(kernel_shape)}."
    ))
  }
  n_spatial <- n_axes - 2L

  strides <- assert_int_param(window_strides)
  x_dil <- assert_int_param(x_dilation)
  kernel_dil <- assert_int_param(kernel_dilation)
  fg_count <- assert_int_param(feature_group_count, len = 1L)
  bg_count <- assert_int_param(batch_group_count, len = 1L)
  assert_choice_param(precision, c("default", "high", "highest"))
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
  pad <- matrix(assert_int_param(as.vector(padding), "padding"), nrow = n_spatial, ncol = 2L)

  # (C2) - (C9)
  per_axis <- list(window_strides = strides, x_dilation = x_dil, kernel_dilation = kernel_dil)
  for (nm in names(per_axis)) {
    assert_entry_per_axis(per_axis[[nm]], n_spatial, nm, axes = "spatial axis")
    if (any(per_axis[[nm]] <= 0L)) {
      cli_abort(c(
        "{.arg {nm}} must be positive.",
        x = "Got {value_repr(per_axis[[nm]])}."
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
  spatial_axes <- list(
    x_spatial_axes = x_spatial_axes,
    kernel_spatial_axes = kernel_spatial_axes,
    output_spatial_axes = output_spatial_axes
  )
  for (nm in names(spatial_axes)) {
    val <- assert_int_param(spatial_axes[[nm]], nm)
    assert_entry_per_axis(val, n_spatial, nm, axes = "spatial axis")
  }

  # (C13), (C18), (C20)
  assert_axis_layout(
    list(
      x_batch_axis = x_batch_axis,
      x_spatial_axes = x_spatial_axes,
      x_feature_axis = x_feature_axis
    ),
    n_axes,
    "x"
  )
  assert_axis_layout(
    list(
      kernel_spatial_axes = kernel_spatial_axes,
      kernel_input_feature_axis = kernel_input_feature_axis,
      kernel_output_feature_axis = kernel_output_feature_axis
    ),
    n_axes,
    "kernel"
  )
  assert_axis_layout(
    list(
      output_batch_axis = output_batch_axis,
      output_spatial_axes = output_spatial_axes,
      output_feature_axis = output_feature_axis
    ),
    n_axes,
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
  assert_same_dtype(x, kernel)

  # (C25), (C26) The window arithmetic runs in double to avoid integer overflow.
  result_shape <- double(n_axes)
  result_shape[output_batch_axis] <- x_batch_size %/% bg_count
  result_shape[output_feature_axis] <- kernel_out_size
  for (sd in seq_len(n_spatial)) {
    x_size <- as.double(x_shape[[x_spatial_axes[[sd]]]])
    k_size <- as.double(kernel_shape[[kernel_spatial_axes[[sd]]]])
    dilated_input <- if (x_size == 0) 0 else (x_size - 1) * x_dil[[sd]] + 1
    padded_input <- pad[sd, 1L] + dilated_input + pad[sd, 2L]

    # XLA `CHECK`-fails (aborting the process) on a negative padded extent.
    if (padded_input < 0) {
      cli_abort(c(
        "Negative {.arg padding} must not remove more than spatial axis {sd} of {.arg x} holds.",
        x = "Axis {x_spatial_axes[[sd]]} of {.arg x} dilates to {dilated_input}, and padding {pad[sd, 1L]} and {pad[sd, 2L]} leaves {padded_input}.", # nolint
        i = "Got {params_repr(list(padding = padding, x_dilation = x_dil))}."
      ))
    }

    # A zero-wide window would otherwise pass the arithmetic below.
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
  assert_flag_param(left_side)
  assert_flag_param(lower)
  assert_flag_param(unit_diagonal)
  assert_flag_param(transpose_a)
  # (I1), (I2)
  assert_array_dtype(a, "float")
  assert_array_dtype(b, "float")
  # (C1)
  assert_same_dtype(a, b)

  shape_a <- shape(a)
  shape_b <- shape(b)
  n_axes <- length(shape_a)

  # (C2)
  if (n_axes < 2L) {
    cli_abort(c(
      "{.arg a} must have at least two axes.",
      x = "Got shape {shape_repr(shape_a)}."
    ))
  }
  if (n_axes != length(shape_b)) {
    cli_abort(c(
      "{.arg a} and {.arg b} must have the same number of axes.",
      x = "Got {shape_repr(shape_a)} and {shape_repr(shape_b)}."
    ))
  }

  # (C3)
  if (shape_a[[n_axes]] != shape_a[[n_axes - 1L]]) {
    cli_abort(c(
      "{.arg a} must be square in its last two axes.",
      x = "Got {shape_repr(shape_a)}."
    ))
  }

  batch_a <- shape_a[seq_len(n_axes - 2L)]
  batch_b <- shape_b[seq_len(n_axes - 2L)]
  if (!identical(batch_a, batch_b)) {
    cli_abort(c(
      "The batch axes of {.arg a} and {.arg b} must match.",
      x = "Got {shape_repr(batch_a)} and {shape_repr(batch_b)}."
    ))
  }

  # (C3)
  side <- if (left_side) shape_b[[n_axes - 1L]] else shape_b[[n_axes]]
  if (shape_a[[n_axes]] != side) {
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

  assert_choice_param(rng_algorithm, c("DEFAULT", "THREE_FRY", "PHILOX"))

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

  out_dtype <- assert_dtype_param(dtype, categories = c("int", "uint", "float"))

  # (C1)
  list(
    state = AbstractArray(dtype = "ui64", shape = state$shape),
    values = AbstractArray(dtype = out_dtype, shape = Shape(assert_shapevec(shape)))
  )
}

infer_cholesky <- function(x, lower) {
  assert_array(x)
  # (I1), (C2), (C3)
  assert_linalg_matrix(x, "x", square = TRUE, batched = TRUE)
  assert_flag_param(lower)

  # (C1)
  list(AbstractArray(dtype = dtype(x), shape = x$shape))
}

infer_qr <- function(x) {
  assert_array(x)
  assert_linalg_matrix(x, "x")
  m <- shape(x)[[1L]]
  n <- shape(x)[[2L]]
  k <- min(m, n)
  list(
    Q = AbstractArray(dtype = dtype(x), shape = Shape(c(m, k))),
    R = AbstractArray(dtype = dtype(x), shape = Shape(c(k, n)))
  )
}

infer_lu <- function(x) {
  assert_array(x)
  assert_linalg_matrix(x, "x")
  m <- shape(x)[[1L]]
  n <- shape(x)[[2L]]
  list(
    LU = AbstractArray(dtype = dtype(x), shape = Shape(c(m, n))),
    pivots = AbstractArray(dtype = default_int(), shape = Shape(min(m, n))),
    permutation = AbstractArray(dtype = default_int(), shape = Shape(m))
  )
}

infer_svd <- function(x) {
  assert_array(x)
  assert_linalg_matrix(x, "x")
  m <- shape(x)[[1L]]
  n <- shape(x)[[2L]]
  k <- min(m, n)
  list(
    d = AbstractArray(dtype = dtype(x), shape = Shape(k)),
    u = AbstractArray(dtype = dtype(x), shape = Shape(c(m, k))),
    vt = AbstractArray(dtype = dtype(x), shape = Shape(c(k, n)))
  )
}

# Names and order mirror `base::eigen()`.
infer_eigh <- function(x) {
  assert_array(x)
  assert_linalg_matrix(x, "x", square = TRUE)
  n <- shape(x)[[1L]]
  list(
    values = AbstractArray(dtype = dtype(x), shape = Shape(n)),
    vectors = AbstractArray(dtype = dtype(x), shape = Shape(c(n, n)))
  )
}

# ---------------------------------------------------------------------------
# Control flow
#
# Branches and loop bodies are already-traced sub-graphs, so their types are
# read off their input and output nodes.
# ---------------------------------------------------------------------------

graph_output_avals <- function(graph) {
  lapply(graph$outputs, function(out) out$aval)
}

# Indices at which two lists of avals disagree in type.
type_mismatches <- function(a, b) {
  which(!vapply(seq_along(a), function(i) eq_type(a[[i]], b[[i]]), logical(1L)))
}

infer_cond <- function(pred, true, false) {
  assert_array_dtype(pred, "bool", shape = integer())
  outs_true <- graph_output_avals(true)
  outs_false <- graph_output_avals(false)

  if (length(outs_true) != length(outs_false)) {
    cli_abort(c(
      "{.arg true} and {.arg false} must return the same number of values.",
      x = "Got {length(outs_true)} and {length(outs_false)}."
    ))
  }
  bad <- type_mismatches(outs_true, outs_false)
  if (length(bad)) {
    described <- sprintf(
      "value %d is %s in `true` and %s in `false`",
      bad,
      vapply(outs_true[bad], repr, character(1L)),
      vapply(outs_false[bad], repr, character(1L))
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
  outs_body <- graph_output_avals(body)
  inputs_body <- lapply(body$inputs, function(inp) inp$aval)
  # The names of `init`, read off the body's input tree rather than passed as a
  # param (which would reach the lowering rules).
  state_names <- pjrt::tree_child_names(body$in_tree)
  labels <- if (length(state_names) == length(outs)) {
    state_names
  } else {
    sprintf("state %d", seq_along(outs))
  }
  describe <- function(idx, a, b, verbs) {
    sprintf(
      "`%s` %s %s and %s %s",
      labels[idx],
      verbs[[1L]],
      vapply(a[idx], repr, character(1L)),
      verbs[[2L]],
      vapply(b[idx], repr, character(1L))
    )
  }

  bad <- type_mismatches(outs, outs_body)
  if (length(bad)) {
    described <- describe(bad, outs, outs_body, c("enters as", "comes back as"))
    # The usual cause is an R value in `init` that materialized at its default.
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
  bad <- type_mismatches(inputs_body, outs_body)
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

  outs_body
}
