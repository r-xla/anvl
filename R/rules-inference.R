# Anvl-native type inference for elementwise primitives.
#
# These helpers take `AbstractArray`(s) directly and return a single
# `AbstractArray` -- no `stablehlo::ValueType` involved, no ambiguity
# handling (that's owned by `make_binary_op()` / `make_unary_op()` in
# `primitives.R`).

# Assert `x` (an `AbstractArray`) has a dtype inheriting from one of the
# given tengen dtype classes (e.g. "FloatType", "IntegerType").
assert_dtype_class <- function(x, ..., arg = rlang::caller_arg(x)) {
  classes <- c(...)
  spec <- list(
    FloatType = list(pred = is_dtype_float, label = "float"),
    IntegerType = list(pred = is_dtype_int, label = "int"),
    UIntegerType = list(pred = is_dtype_uint, label = "uint"),
    BooleanType = list(pred = is_dtype_bool, label = "bool")
  )
  dt <- dtype(x)
  ok <- any(vapply(classes, function(cl) spec[[cl]]$pred(dt), logical(1L)))
  if (!ok) {
    labels <- vapply(classes, function(cl) spec[[cl]]$label, character(1L))
    cli_abort(c(
      "{.arg {arg}} must have dtype {.or {labels}}.",
      x = "Got {.val {as.character(dt)}}."
    ))
  }
  invisible(x)
}

# Assert two `AbstractArray`s have identical dtype and shape.
assert_same_type <- function(a, b, arg_a = "lhs", arg_b = "rhs") {
  if (dtype(a) != dtype(b)) {
    cli_abort(c(
      "{.arg {arg_a}} and {.arg {arg_b}} must have the same dtype.",
      x = "Got {.val {as.character(dtype(a))}} and {.val {as.character(dtype(b))}}."
    ))
  }
  if (!identical(shape(a), shape(b))) {
    cli_abort(c(
      "{.arg {arg_a}} and {.arg {arg_b}} must have the same shape.",
      x = "Got {xlamisc::shapevec_repr(shape(a))} and {xlamisc::shapevec_repr(shape(b))}."
    ))
  }
  invisible(NULL)
}

# Infer for elementwise binary operations with no dtype restriction.
infer_generic_biv <- function(lhs, rhs) {
  assert_same_type(lhs, rhs)
  lhs
}

# Infer for elementwise float binary operations.
infer_float_biv <- function(lhs, rhs) {
  assert_same_type(lhs, rhs)
  assert_dtype_class(lhs, "FloatType")
  lhs
}

# Infer for elementwise integerish (boolean or integer) binary operations.
infer_integerish_biv <- function(lhs, rhs) {
  assert_dtype_class(lhs, "BooleanType", "IntegerType", "UIntegerType")
  assert_dtype_class(rhs, "BooleanType", "IntegerType", "UIntegerType")
  assert_same_type(lhs, rhs)
  lhs
}

# Infer for elementwise numeric (float or integer) binary operations.
infer_numeric_biv <- function(lhs, rhs) {
  assert_dtype_class(lhs, "FloatType", "IntegerType", "UIntegerType")
  assert_same_type(lhs, rhs)
  lhs
}

# Infer for elementwise integerish (boolean or integer) unary operations.
infer_integerish_uni <- function(x) {
  assert_dtype_class(x, "BooleanType", "IntegerType", "UIntegerType")
  x
}

# Infer for elementwise float unary operations.
infer_float_uni <- function(x) {
  assert_dtype_class(x, "FloatType")
  x
}

# Infer for elementwise integer (signed or unsigned) unary operations.
infer_integer_uni <- function(x) {
  assert_dtype_class(x, "IntegerType", "UIntegerType")
  x
}

# Infer for elementwise numeric (float or integer) unary operations.
infer_numeric_uni <- function(x) {
  assert_dtype_class(x, "FloatType", "IntegerType", "UIntegerType")
  x
}

# --- Custom elementwise / shape-preserving ops -------------------------------

# abs / sign: float or (signed) integer, type preserved.
infer_abs <- function(x) {
  assert_dtype_class(x, "FloatType", "IntegerType")
  x
}
infer_sign <- function(x) {
  assert_dtype_class(x, "FloatType", "IntegerType")
  x
}

# is_finite: float input, boolean output of the same shape.
infer_is_finite <- function(x) {
  assert_dtype_class(x, "FloatType")
  AbstractArray("bool", shape(x))
}

# polygamma(n, x): float, same shape/dtype for both args.
infer_polygamma <- function(n, x) {
  infer_float_biv(n, x)
}

# clamp(min, x, max): `min`/`max` must share `x`'s dtype and either match its
# shape or be scalars. Result has `x`'s type.
infer_clamp <- function(min, x, max) {
  if (dtype(min) != dtype(x) || dtype(max) != dtype(x)) {
    cli_abort(c(
      "{.arg min}, {.arg x}, and {.arg max} must have the same dtype.",
      x = "Got {.val {as.character(dtype(min))}}, {.val {as.character(dtype(x))}}, and {.val {as.character(dtype(max))}}."
    ))
  }
  xs <- shape(x)
  if (naxes(min) != 0L && !identical(shape(min), xs)) {
    cli_abort(c(
      "{.arg min} must have the same shape as {.arg x} or be a scalar.",
      x = "Got shapes {xlamisc::shapevec_repr(shape(min))} and {xlamisc::shapevec_repr(xs)}."
    ))
  }
  if (naxes(max) != 0L && !identical(shape(max), xs)) {
    cli_abort(c(
      "{.arg max} must have the same shape as {.arg x} or be a scalar.",
      x = "Got shapes {xlamisc::shapevec_repr(shape(max))} and {xlamisc::shapevec_repr(xs)}."
    ))
  }
  x
}

# compare(lhs, rhs, comparison_direction): equal-typed inputs, boolean output.
infer_compare <- function(lhs, rhs, comparison_direction) {
  valid <- c("EQ", "NE", "GE", "GT", "LE", "LT")
  if (!(comparison_direction %in% valid)) {
    cli_abort(c(
      "{.arg comparison_direction} must be one of {.val {valid}}.",
      x = "Got {.val {comparison_direction}}."
    ))
  }
  assert_same_type(lhs, rhs)
  AbstractArray("bool", shape(lhs))
}

# select(pred, true_value, false_value): branches must share a type; `pred` is
# boolean and either a scalar or the same shape as the branches.
infer_select <- function(pred, true_value, false_value) {
  assert_same_type(true_value, false_value, "true_value", "false_value")
  assert_dtype_class(pred, "BooleanType")
  if (naxes(pred) != 0L && !identical(shape(pred), shape(true_value))) {
    cli_abort(c(
      "{.arg pred} must be a scalar or have the same shape as {.arg true_value}.",
      x = "Got shapes {xlamisc::shapevec_repr(shape(pred))} and {xlamisc::shapevec_repr(shape(true_value))}."
    ))
  }
  true_value
}

# bitcast_convert(x, dtype): reinterpret the bytes of `x` as `dtype`. When the
# target is wider, the last axis (whose size must equal the width ratio) is
# consumed; when narrower, a trailing axis is appended.
infer_bitcast_convert <- function(x, dtype) {
  dtype <- as.character(dtype)
  if (is_dtype_bool(dtype(x))) {
    cli_abort(c(
      "Bitcast conversions from and to i1 are not supported.",
      x = "{.arg x} has dtype {.val {as.character(dtype(x))}}."
    ))
  }
  if (dtype %in% c("i1", "pred", "bool")) {
    cli_abort(c(
      "Bitcast conversions from and to i1 are not supported.",
      x = "{.arg dtype} is {.val {as.character(as_dtype(dtype))}}."
    ))
  }
  supported <- c(
    "i8",
    "i16",
    "i32",
    "i64",
    "ui8",
    "ui16",
    "ui32",
    "ui64",
    "f8",
    "f16",
    "f32",
    "f64"
  )
  if (!(dtype %in% supported)) {
    cli_abort("Unsupported dtype: {dtype}")
  }
  bits <- function(dt) as.integer(sub(".*?([0-9]+)$", "\\1", as.character(dt)))
  cst_fct <- bits(dtype) / bits(dtype(x))
  shp <- shape(x)
  if (cst_fct == 1) {
    result_shape <- shp
  } else if (cst_fct > 1) {
    if (length(shp) == 0L) {
      cli_abort(c(
        "{.arg x} must have at least one axis for this bitcast conversion.",
        x = "{.arg x} is a scalar ({.val {as.character(dtype(x))}} -> {.val {dtype}})."
      ))
    } else if (shp[[length(shp)]] != cst_fct) {
      cli_abort(c(
        "The last axis of {.arg x} must have size {cst_fct} for this bitcast conversion.",
        x = "Got size {shp[[length(shp)]]}."
      ))
    } else {
      result_shape <- shp[seq_len(length(shp) - 1L)]
    }
  } else {
    result_shape <- c(shp, as.integer(1 / cst_fct))
  }
  AbstractArray(dtype, result_shape)
}

# --- Shape-changing ops ------------------------------------------------------

# transpose(x, permutation): permute the axes of `x`. `permutation` is a
# 1-based permutation of the axes of `x`.
infer_transpose <- function(x, permutation) {
  rank <- naxes(x)
  if (!setequal(permutation, seq_len(rank))) {
    cli_abort(c(
      "{.arg permutation} must be a permutation of the axes {.val {seq_len(rank)}}.",
      x = "Got {.val {permutation}}."
    ))
  }
  AbstractArray(dtype(x), shape(x)[permutation])
}

# reshape(x, shape): reinterpret `x` with a new shape holding the same number
# of elements.
infer_reshape <- function(x, shape) {
  result_dims <- as.integer(shape)
  if (prod(shape(x)) != prod(result_dims)) {
    cli_abort(c(
      "The output must have the same number of elements as {.arg x}.",
      x = "Got input shape {xlamisc::shapevec_repr(shape(x))} and output shape {xlamisc::shapevec_repr(result_dims)}."
    ))
  }
  AbstractArray(dtype(x), result_dims)
}

# broadcast_in_axes(x, shape, broadcast_axes): broadcast `x` to `shape`.
# `broadcast_axes[d]` is the (1-based) axis of the output that axis `d` of `x`
# maps onto.
infer_broadcast_in_axes <- function(x, shape, broadcast_axes) {
  operand_dims <- shape(x)
  result_dims <- as.integer(shape)
  bdims <- as.integer(broadcast_axes)
  if (length(bdims) != length(operand_dims)) {
    cli_abort(c(
      "{.arg broadcast_axes} must have one entry per axis of {.arg x}.",
      x = "Got {length(bdims)} entr{?y/ies} for {.arg x} with {length(operand_dims)} ax{?is/es}."
    ))
  }
  if (any(bdims < 1L | bdims > length(result_dims))) {
    cli_abort(c(
      "{.arg broadcast_axes} must be between 1 and {length(result_dims)}.",
      x = "Got {.val {bdims}}."
    ))
  }
  if (anyDuplicated(bdims)) {
    cli_abort(c(
      "{.arg broadcast_axes} must not contain duplicate axes.",
      x = "Got {.val {bdims}}."
    ))
  }
  for (d in seq_along(bdims)) {
    op_dim <- operand_dims[d]
    out_dim <- result_dims[bdims[d]]
    if (op_dim != out_dim && op_dim != 1L) {
      cli_abort(c(
        "Cannot broadcast axis {d} of {.arg x} onto output axis {bdims[d]}.",
        x = "Axis {d} of {.arg x} has size {op_dim}, but output axis {bdims[d]} has size {out_dim}."
      ))
    }
  }
  AbstractArray(dtype(x), result_dims)
}

# reverse(x, axes): reverse the order of elements along `axes`. Shape is
# unchanged. `axes` is validated (bounds, uniqueness) upstream by
# `resolve_axes()`.
infer_reverse <- function(x, axes) {
  if (length(axes) == 0L) {
    cli_abort("{.arg axes} must contain at least one axis.")
  }
  AbstractArray(dtype(x), shape(x))
}

# iota(axis, dtype, shape, ...): build an increasing sequence along `axis`
# (1-based) of the given `shape`.
infer_iota <- function(axis, dtype, shape, start, ambiguous) {
  shape <- as.integer(shape)
  num_axes <- length(shape)
  if (axis < 1L || axis > num_axes) {
    cli_abort(c(
      "{.arg axis} must be between 1 and {num_axes}.",
      x = "Got {.val {axis}}."
    ))
  }
  dt <- as_dtype(dtype)
  if (!(is_dtype_int(dt) || is_dtype_uint(dt) || is_dtype_float(dt))) {
    cli_abort(c(
      "{.arg dtype} must be an integer, unsigned integer, or floating-point type.",
      x = "Got {.val {as.character(dt)}}."
    ))
  }
  list(IotaArray(shape = shape, dtype = dtype, axis = axis, start = start, ambiguous = ambiguous))
}

# concatenate(xs, axis): join the arrays in `xs` along `axis` (1-based). All
# inputs must share dtype and shape except along `axis`.
infer_concatenate <- function(xs, axis) {
  dtypes <- vapply(xs, function(x) as.character(dtype(x)), character(1L))
  if (length(unique(dtypes)) != 1L) {
    cli_abort(c(
      "All arrays to concatenate must have the same dtype.",
      x = "Got {.val {dtypes}}."
    ))
  }
  input_dims <- lapply(xs, shape)
  num_axes <- length(input_dims[[1L]])
  if (axis < 1L || axis > num_axes) {
    cli_abort(c(
      "{.arg axis} must be between 1 and {num_axes}.",
      x = "Got {.val {axis}}."
    ))
  }
  dims_no_concat <- lapply(input_dims, function(d) d[-axis])
  if (!all(vapply(dims_no_concat, identical, logical(1L), dims_no_concat[[1L]]))) {
    cli_abort(c(
      "All arrays to concatenate must have the same shape, except along {.arg axis} ({axis}).",
      x = "Got shapes {.val {vapply(input_dims, xlamisc::shapevec_repr, character(1L))}}."
    ))
  }
  result_dims <- input_dims[[1L]]
  result_dims[axis] <- sum(vapply(input_dims, function(d) d[axis], integer(1L)))
  AbstractArray(dtype(xs[[1L]]), result_dims)
}

# slice(x, start_indices, limit_indices, strides): extract a strided slice.
# `start_indices` / `limit_indices` are 1-based and inclusive; the extent of
# each axis is `ceiling((limit - start + 1) / stride)`.
infer_slice <- function(x, start_indices, limit_indices, strides) {
  operand_shape <- shape(x)
  rank <- length(operand_shape)
  start_idx <- as.integer(start_indices)
  limit_idx <- as.integer(limit_indices)
  stride_vals <- as.integer(strides)
  if (length(start_idx) != length(limit_idx) || length(start_idx) != length(stride_vals)) {
    cli_abort(c(
      "{.arg start_indices}, {.arg limit_indices}, and {.arg strides} must have the same length.",
      x = "Got lengths {length(start_idx)}, {length(limit_idx)}, and {length(stride_vals)}."
    ))
  }
  if (length(start_idx) != rank) {
    cli_abort(c(
      "{.arg start_indices}, {.arg limit_indices}, and {.arg strides} must have one entry per axis of {.arg x} ({rank}).",
      x = "Got length {length(start_idx)}."
    ))
  }
  if (any(start_idx < 1L)) {
    cli_abort(c(
      "{.arg start_indices} must be at least 1.",
      x = "Got {.val {start_idx}}."
    ))
  }
  if (any(start_idx > limit_idx)) {
    cli_abort(c(
      "{.arg start_indices} must not exceed {.arg limit_indices}.",
      x = "Got start {.val {start_idx}} and limit {.val {limit_idx}}."
    ))
  }
  if (any(limit_idx > operand_shape)) {
    cli_abort(c(
      "{.arg limit_indices} must not exceed the size of {.arg x} along each axis.",
      x = "Got limit {.val {limit_idx}} and shape {xlamisc::shapevec_repr(operand_shape)}."
    ))
  }
  if (any(stride_vals < 1L)) {
    cli_abort(c(
      "{.arg strides} must be at least 1.",
      x = "Got {.val {stride_vals}}."
    ))
  }
  result_dims <- as.integer(ceiling((limit_idx - start_idx + 1L) / stride_vals))
  AbstractArray(dtype(x), result_dims)
}

# pad(x, padding_value, edge_padding_low, edge_padding_high, interior_padding):
# pad `x` with `padding_value`. The padding vectors give per-axis element
# counts (not indices), so no 1-based conversion applies.
infer_pad <- function(x, padding_value, edge_padding_low, edge_padding_high, interior_padding) {
  if (naxes(padding_value) != 0L) {
    cli_abort(c(
      "{.arg padding_value} must be a scalar.",
      x = "Got shape {xlamisc::shapevec_repr(shape(padding_value))}."
    ))
  }
  if (dtype(x) != dtype(padding_value)) {
    cli_abort(c(
      "{.arg x} and {.arg padding_value} must have the same dtype.",
      x = "Got {.val {as.character(dtype(x))}} and {.val {as.character(dtype(padding_value))}}."
    ))
  }
  operand_shape <- shape(x)
  rank <- length(operand_shape)
  low <- as.integer(edge_padding_low)
  high <- as.integer(edge_padding_high)
  interior <- as.integer(interior_padding)
  check <- function(val, name) {
    if (length(val) != rank) {
      cli_abort(c(
        "{name} must have one entry per axis of {.arg x} ({rank}).",
        x = "Got length {length(val)}."
      ))
    }
  }
  check(low, "edge_padding_low")
  check(high, "edge_padding_high")
  check(interior, "interior_padding")
  if (any(interior < 0L)) {
    cli_abort(c(
      "{.arg interior_padding} must be non-negative.",
      x = "Got {.val {interior}}."
    ))
  }
  lowhigh <- rbind(low, high)
  lowhigh[lowhigh > 0] <- 0
  if (any(colSums(abs(lowhigh)) > rank)) {
    cli_abort(c(
      "Negative padding must not exceed the number of axes of {.arg x}.",
      x = "edge_padding_low: {.val {low}}, edge_padding_high: {.val {high}}, axes: {rank}."
    ))
  }
  result_shape <- as.integer(
    operand_shape + low + pmax(operand_shape - 1L, 0L) * interior + high
  )
  AbstractArray(dtype(x), result_shape)
}

# top_k(x, k): the `k` largest values along the last axis and their indices.
infer_top_k <- function(x, k) {
  assert_dtype_class(x, "FloatType", "IntegerType", "UIntegerType")
  operand_shape <- shape(x)
  rank <- length(operand_shape)
  if (rank < 1L) {
    cli_abort(c(
      "{.arg x} must have at least one axis.",
      x = "Got a scalar."
    ))
  }
  last_dim <- operand_shape[[rank]]
  if (k > last_dim) {
    cli_abort(c(
      "{.arg k} must not exceed the size of the last axis of {.arg x}.",
      x = "Got k = {.val {k}} and last axis size {.val {last_dim}}."
    ))
  }
  result_shape <- operand_shape
  result_shape[[rank]] <- as.integer(k)
  list(
    AbstractArray(dtype(x), result_shape),
    AbstractArray("i32", result_shape)
  )
}

# triangular_solve(a, b, ...): solve a triangular system. Only shape/dtype
# validation; no index math.
infer_triangular_solve <- function(a, b, left_side, lower, unit_diagonal, transpose_a) {
  if (dtype(a) != dtype(b)) {
    cli_abort(c(
      "{.arg a} and {.arg b} must have the same dtype.",
      x = "Got {.val {as.character(dtype(a))}} and {.val {as.character(dtype(b))}}."
    ))
  }
  a_dims <- shape(a)
  b_dims <- shape(b)
  rank_a <- length(a_dims)
  rank_b <- length(b_dims)
  if (rank_a < 2L) {
    cli_abort(c(
      "{.arg a} must have at least 2 axes.",
      x = "Got {rank_a}."
    ))
  }
  if (rank_a != rank_b) {
    cli_abort(c(
      "{.arg a} and {.arg b} must have the same number of axes.",
      x = "Got {rank_a} and {rank_b}."
    ))
  }
  if (a_dims[rank_a] != a_dims[rank_a - 1L]) {
    cli_abort(c(
      "{.arg a} must be a square matrix (its last two axes must have equal size).",
      x = "Got shape {xlamisc::shapevec_repr(a_dims)}."
    ))
  }
  if (rank_a > 2L) {
    a_batch <- a_dims[seq_len(rank_a - 2L)]
    b_batch <- b_dims[seq_len(rank_b - 2L)]
    if (!identical(a_batch, b_batch)) {
      cli_abort(c(
        "The batch axes of {.arg a} and {.arg b} must match.",
        x = "Got {xlamisc::shapevec_repr(a_batch)} and {xlamisc::shapevec_repr(b_batch)}."
      ))
    }
  }
  a_size <- a_dims[rank_a]
  b_relevant_dim <- if (left_side) b_dims[rank_b - 1L] else b_dims[rank_b]
  if (a_size != b_relevant_dim) {
    cli_abort(c(
      "Size mismatch between {.arg a} and {.arg b}.",
      x = "Got shapes {xlamisc::shapevec_repr(a_dims)} and {xlamisc::shapevec_repr(b_dims)}."
    ))
  }
  AbstractArray(dtype(b), b_dims)
}

# rng_bit_generator(initial_state, rng_algorithm, dtype, shape): returns the
# updated RNG state (same ui64 shape as `initial_state`) and the generated
# values of the given `dtype` / `shape`.
infer_rng_bit_generator <- function(initial_state, rng_algorithm, dtype, shape) {
  if (as.character(dtype(initial_state)) != "ui64") {
    cli_abort(c(
      "{.arg initial_state} must have dtype ui64.",
      x = "Got {.val {as.character(dtype(initial_state))}}."
    ))
  }
  if (naxes(initial_state) != 1L) {
    cli_abort(c(
      "{.arg initial_state} must have exactly one axis.",
      x = "Got shape {xlamisc::shapevec_repr(shape(initial_state))}."
    ))
  }
  valid_algos <- c("DEFAULT", "THREE_FRY", "PHILOX")
  if (!(rng_algorithm %in% valid_algos)) {
    cli_abort(c(
      "{.arg rng_algorithm} must be one of {.val {valid_algos}}.",
      x = "Got {.val {rng_algorithm}}."
    ))
  }
  init_shape <- shape(initial_state)
  state_size <- init_shape[[1L]]
  if (rng_algorithm == "THREE_FRY" && state_size != 2L) {
    cli_abort(c(
      "{.val THREE_FRY} requires an initial state of length 2.",
      x = "Got {.val {state_size}}."
    ))
  }
  if (rng_algorithm == "PHILOX" && !(state_size %in% c(2L, 3L))) {
    cli_abort(c(
      "{.val PHILOX} requires an initial state of length 2 or 3.",
      x = "Got {.val {state_size}}."
    ))
  }
  out_dtype <- as_dtype(dtype)
  if (!(is_dtype_int(out_dtype) || is_dtype_uint(out_dtype) || is_dtype_float(out_dtype))) {
    cli_abort(c(
      "{.arg dtype} must be an integer, unsigned integer, or floating-point type.",
      x = "Got {.val {as.character(out_dtype)}}."
    ))
  }
  out_shape <- as.integer(shape)
  list(
    AbstractArray("ui64", init_shape),
    AbstractArray(out_dtype, out_shape)
  )
}

# --- DimensionNumbers ops ----------------------------------------------------
#
# gather / scatter / convolution / dot_general carry the densest axis logic.
# All axis-set parameters are 1-based here (validated `1 <= axis <= rank`); the
# output shape is assembled by placing sizes at 1-based positions directly.
# The 0-based `- 1L` conversion for the StableHLO DimensionNumbers structs lives
# only in the lowering rules (`R/rules-stablehlo.R`).

# TRUE if any entry of `axes` is outside the 1-based axis range `1:rank`.
axes_out_of_range <- function(axes, rank) {
  length(axes) > 0L && (any(axes < 1L) || any(axes > rank))
}

# gather(x, start_indices, ...): read slices from `x` at the positions given by
# `start_indices`. The output is assembled from the batch axes (the axes of
# `start_indices` other than `index_vector_axis`) interleaved with the offset
# axes (the non-collapsed, non-batching slice axes of `x`).
infer_gather <- function(
  x,
  start_indices,
  slice_sizes,
  offset_axes,
  collapsed_slice_axes,
  x_batching_axes,
  start_indices_batching_axes,
  start_index_map,
  index_vector_axis
) {
  x_shape <- shape(x)
  x_rank <- naxes(x)
  si_shape <- shape(start_indices)
  si_rank <- naxes(start_indices)
  slice_sizes <- as.integer(slice_sizes)

  # (C1)
  expected_rank <- length(offset_axes) +
    length(collapsed_slice_axes) +
    length(x_batching_axes)
  if (x_rank != expected_rank) {
    cli_abort(c(
      "The number of axes of {.arg x} must equal length(offset_axes) + length(collapsed_slice_axes) + length(x_batching_axes).",
      x = "Got {x_rank}, but expected {expected_rank} (= {length(offset_axes)} + {length(collapsed_slice_axes)} + {length(x_batching_axes)})."
    ))
  }

  # (C2)
  if (index_vector_axis < 1L || index_vector_axis > si_rank + 1L) {
    cli_abort(c(
      "{.arg index_vector_axis} must be between 1 and {si_rank + 1L}.",
      x = "Got {.val {index_vector_axis}}."
    ))
  }

  # (C3)
  expected_start_index_map_size <- if (index_vector_axis <= si_rank) {
    si_shape[index_vector_axis]
  } else {
    1L
  }
  if (length(start_index_map) != expected_start_index_map_size) {
    cli_abort(c(
      "length(start_index_map) must equal the index vector size.",
      x = "Got {length(start_index_map)}, but expected {expected_start_index_map_size}."
    ))
  }

  # (C4)
  if (anyDuplicated(offset_axes)) {
    cli_abort(c(
      "{.arg offset_axes} must not contain duplicate axes.",
      x = "Got {.val {offset_axes}}."
    ))
  }
  if (is.unsorted(offset_axes)) {
    cli_abort(c(
      "{.arg offset_axes} must be sorted in increasing order.",
      x = "Got {.val {offset_axes}}."
    ))
  }

  # Output shape components (for C5 and C22).
  batch_dim_sizes <- if (index_vector_axis == si_rank + 1L) {
    si_shape
  } else {
    without(si_shape, index_vector_axis)
  }
  offset_dim_sizes <- without(
    slice_sizes,
    c(collapsed_slice_axes, x_batching_axes)
  )
  result_rank <- length(batch_dim_sizes) + length(offset_dim_sizes)

  # (C5)
  if (axes_out_of_range(offset_axes, result_rank)) {
    cli_abort(c(
      "{.arg offset_axes} must be between 1 and {result_rank}.",
      x = "Got {.val {offset_axes}}."
    ))
  }

  # (C6)
  if (anyDuplicated(c(collapsed_slice_axes, x_batching_axes))) {
    cli_abort(c(
      "{.arg collapsed_slice_axes} and {.arg x_batching_axes} must not share axes or contain duplicates.",
      x = "Got {.val {collapsed_slice_axes}} and {.val {x_batching_axes}}."
    ))
  }

  # (C7)
  if (is.unsorted(collapsed_slice_axes)) {
    cli_abort(c(
      "{.arg collapsed_slice_axes} must be sorted in increasing order.",
      x = "Got {.val {collapsed_slice_axes}}."
    ))
  }

  # (C8)
  if (axes_out_of_range(collapsed_slice_axes, x_rank)) {
    cli_abort(c(
      "{.arg collapsed_slice_axes} must be between 1 and {x_rank}.",
      x = "Got {.val {collapsed_slice_axes}}."
    ))
  }

  # (C9)
  if (length(collapsed_slice_axes)) {
    collapsed_sizes <- slice_sizes[collapsed_slice_axes]
    if (any(collapsed_sizes > 1L)) {
      cli_abort(c(
        "{.arg slice_sizes} at {.arg collapsed_slice_axes} must be <= 1.",
        x = "Got {.val {collapsed_sizes}}."
      ))
    }
  }

  # (C10)
  if (is.unsorted(x_batching_axes)) {
    cli_abort(c(
      "{.arg x_batching_axes} must be sorted in increasing order.",
      x = "Got {.val {x_batching_axes}}."
    ))
  }

  # (C11)
  if (axes_out_of_range(x_batching_axes, x_rank)) {
    cli_abort(c(
      "{.arg x_batching_axes} must be between 1 and {x_rank}.",
      x = "Got {.val {x_batching_axes}}."
    ))
  }

  # (C12)
  if (length(x_batching_axes)) {
    batching_sizes <- slice_sizes[x_batching_axes]
    if (any(batching_sizes > 1L)) {
      cli_abort(c(
        "{.arg slice_sizes} at {.arg x_batching_axes} must be <= 1.",
        x = "Got {.val {batching_sizes}}."
      ))
    }
  }

  # (C13)
  if (anyDuplicated(start_indices_batching_axes)) {
    cli_abort(c(
      "{.arg start_indices_batching_axes} must not contain duplicate axes.",
      x = "Got {.val {start_indices_batching_axes}}."
    ))
  }

  # (C14)
  if (axes_out_of_range(start_indices_batching_axes, si_rank)) {
    cli_abort(c(
      "{.arg start_indices_batching_axes} must be between 1 and {si_rank}.",
      x = "Got {.val {start_indices_batching_axes}}."
    ))
  }

  # (C15)
  if (index_vector_axis %in% start_indices_batching_axes) {
    cli_abort(c(
      "{.arg index_vector_axis} must not be in {.arg start_indices_batching_axes}.",
      x = "Got index_vector_axis = {.val {index_vector_axis}} and start_indices_batching_axes = {.val {start_indices_batching_axes}}."
    ))
  }

  # (C16)
  if (length(x_batching_axes) != length(start_indices_batching_axes)) {
    cli_abort(c(
      "{.arg x_batching_axes} and {.arg start_indices_batching_axes} must have the same length.",
      x = "Got {length(x_batching_axes)} and {length(start_indices_batching_axes)}."
    ))
  }

  # (C17)
  if (length(x_batching_axes)) {
    batch_shape_x <- x_shape[x_batching_axes]
    batch_shape_si <- si_shape[start_indices_batching_axes]
    if (!identical(batch_shape_x, batch_shape_si)) {
      cli_abort(c(
        "The batch axes of {.arg x} and {.arg start_indices} must have matching sizes.",
        x = "Got {xlamisc::shapevec_repr(batch_shape_x)} and {xlamisc::shapevec_repr(batch_shape_si)}."
      ))
    }
  }

  # (C18)
  if (anyDuplicated(c(start_index_map, x_batching_axes))) {
    cli_abort(c(
      "{.arg start_index_map} and {.arg x_batching_axes} must not share axes or contain duplicates.",
      x = "Got {.val {start_index_map}} and {.val {x_batching_axes}}."
    ))
  }

  # (C19)
  if (axes_out_of_range(start_index_map, x_rank)) {
    cli_abort(c(
      "{.arg start_index_map} must be between 1 and {x_rank}.",
      x = "Got {.val {start_index_map}}."
    ))
  }

  # (C20)
  if (length(slice_sizes) != x_rank) {
    cli_abort(c(
      "length(slice_sizes) must equal the number of axes of {.arg x} ({x_rank}).",
      x = "Got {length(slice_sizes)}."
    ))
  }

  # (C21)
  if (any(slice_sizes < 0L) || any(slice_sizes > x_shape)) {
    cli_abort(c(
      "{.arg slice_sizes} must satisfy 0 <= slice_sizes <= shape(x).",
      x = "Got {.val {slice_sizes}}, but {.arg x} has shape {xlamisc::shapevec_repr(x_shape)}."
    ))
  }

  # (C22)
  batch_axes <- setdiff(seq_len(result_rank), offset_axes)
  result_shape <- integer(result_rank)
  if (length(batch_axes) > 0L) {
    result_shape[batch_axes] <- batch_dim_sizes
  }
  if (length(offset_axes) > 0L) {
    result_shape[offset_axes] <- offset_dim_sizes
  }

  AbstractArray(dtype(x), result_shape)
}

# scatter(x, scatter_indices, update, ...): write `update` slices into `x` at
# the positions given by `scatter_indices`. The result has the same shape and
# dtype as `x`; the axis sets partition the `update` and `x` axes. anvl only
# supports a single input / update.
infer_scatter <- function(
  x,
  scatter_indices,
  update,
  update_window_axes,
  inserted_window_axes,
  x_batching_axes,
  scatter_indices_batching_axes,
  scatter_axes_to_x_axes,
  index_vector_axis
) {
  x_shape <- shape(x)
  x_rank <- naxes(x)
  si_shape <- shape(scatter_indices)
  si_rank <- naxes(scatter_indices)
  updates_shape <- shape(update)
  updates_rank <- naxes(update)

  # (C6)
  if (dtype(x) != dtype(update)) {
    cli_abort(c(
      "{.arg x} and {.arg update} must have the same dtype.",
      x = "Got {.val {as.character(dtype(x))}} and {.val {as.character(dtype(update))}}."
    ))
  }

  # (C2)
  expected_rank <- length(update_window_axes) +
    length(inserted_window_axes) +
    length(x_batching_axes)
  if (x_rank != expected_rank) {
    cli_abort(c(
      "The number of axes of {.arg x} must equal length(update_window_axes) + length(inserted_window_axes) + length(x_batching_axes).",
      x = "Got {x_rank}, but expected {expected_rank} (= {length(update_window_axes)} + {length(inserted_window_axes)} + {length(x_batching_axes)})."
    ))
  }

  # (C7)
  if (anyDuplicated(update_window_axes)) {
    cli_abort(c(
      "{.arg update_window_axes} must not contain duplicate axes.",
      x = "Got {.val {update_window_axes}}."
    ))
  }
  if (is.unsorted(update_window_axes)) {
    cli_abort(c(
      "{.arg update_window_axes} must be sorted in increasing order.",
      x = "Got {.val {update_window_axes}}."
    ))
  }

  # (C8)
  if (axes_out_of_range(update_window_axes, updates_rank)) {
    cli_abort(c(
      "{.arg update_window_axes} must be between 1 and {updates_rank}.",
      x = "Got {.val {update_window_axes}}."
    ))
  }

  # (C9)
  if (anyDuplicated(c(inserted_window_axes, x_batching_axes))) {
    cli_abort(c(
      "{.arg inserted_window_axes} and {.arg x_batching_axes} must not share axes or contain duplicates.",
      x = "Got {.val {inserted_window_axes}} and {.val {x_batching_axes}}."
    ))
  }

  # (C10)
  if (is.unsorted(inserted_window_axes)) {
    cli_abort(c(
      "{.arg inserted_window_axes} must be sorted in increasing order.",
      x = "Got {.val {inserted_window_axes}}."
    ))
  }

  # (C11)
  if (axes_out_of_range(inserted_window_axes, x_rank)) {
    cli_abort(c(
      "{.arg inserted_window_axes} must be between 1 and {x_rank}.",
      x = "Got {.val {inserted_window_axes}}."
    ))
  }

  # (C12)
  if (is.unsorted(x_batching_axes)) {
    cli_abort(c(
      "{.arg x_batching_axes} must be sorted in increasing order.",
      x = "Got {.val {x_batching_axes}}."
    ))
  }

  # (C13)
  if (axes_out_of_range(x_batching_axes, x_rank)) {
    cli_abort(c(
      "{.arg x_batching_axes} must be between 1 and {x_rank}.",
      x = "Got {.val {x_batching_axes}}."
    ))
  }

  # (C14)
  if (anyDuplicated(scatter_indices_batching_axes)) {
    cli_abort(c(
      "{.arg scatter_indices_batching_axes} must not contain duplicate axes.",
      x = "Got {.val {scatter_indices_batching_axes}}."
    ))
  }

  # (C15)
  if (axes_out_of_range(scatter_indices_batching_axes, si_rank)) {
    cli_abort(c(
      "{.arg scatter_indices_batching_axes} must be between 1 and {si_rank}.",
      x = "Got {.val {scatter_indices_batching_axes}}."
    ))
  }

  # (C16)
  if (index_vector_axis %in% scatter_indices_batching_axes) {
    cli_abort(c(
      "{.arg index_vector_axis} must not be in {.arg scatter_indices_batching_axes}.",
      x = "Got index_vector_axis = {.val {index_vector_axis}} and scatter_indices_batching_axes = {.val {scatter_indices_batching_axes}}."
    ))
  }

  # (C17)
  if (length(x_batching_axes) != length(scatter_indices_batching_axes)) {
    cli_abort(c(
      "{.arg x_batching_axes} and {.arg scatter_indices_batching_axes} must have the same length.",
      x = "Got {length(x_batching_axes)} and {length(scatter_indices_batching_axes)}."
    ))
  }

  # (C18)
  batch_shape_x <- x_shape[x_batching_axes]
  batch_shape_si <- si_shape[scatter_indices_batching_axes]
  if (!identical(batch_shape_x, batch_shape_si)) {
    cli_abort(c(
      "The batch axes of {.arg x} and {.arg scatter_indices} must have matching sizes.",
      x = "Got {xlamisc::shapevec_repr(batch_shape_x)} and {xlamisc::shapevec_repr(batch_shape_si)}."
    ))
  }

  # (C22)
  if (index_vector_axis < 1L || index_vector_axis > si_rank + 1L) {
    cli_abort(c(
      "{.arg index_vector_axis} must be between 1 and {si_rank + 1L}.",
      x = "Got {.val {index_vector_axis}}."
    ))
  }

  # (C19)
  expected_scatter_dims_size <- if (index_vector_axis <= si_rank) {
    si_shape[index_vector_axis]
  } else {
    1L
  }
  if (length(scatter_axes_to_x_axes) != expected_scatter_dims_size) {
    cli_abort(c(
      "length(scatter_axes_to_x_axes) must equal the index vector size.",
      x = "Got {length(scatter_axes_to_x_axes)}, but expected {expected_scatter_dims_size}."
    ))
  }

  # (C20)
  if (anyDuplicated(c(scatter_axes_to_x_axes, x_batching_axes))) {
    cli_abort(c(
      "{.arg scatter_axes_to_x_axes} and {.arg x_batching_axes} must not share axes or contain duplicates.",
      x = "Got {.val {scatter_axes_to_x_axes}} and {.val {x_batching_axes}}."
    ))
  }

  # (C21)
  if (axes_out_of_range(scatter_axes_to_x_axes, x_rank)) {
    cli_abort(c(
      "{.arg scatter_axes_to_x_axes} must be between 1 and {x_rank}.",
      x = "Got {.val {scatter_axes_to_x_axes}}."
    ))
  }

  update_scatter_axes <- setdiff(seq_len(updates_rank), update_window_axes)

  update_scatter_dim_sizes <- if (index_vector_axis == si_rank + 1L) {
    si_shape
  } else {
    without(si_shape, index_vector_axis)
  }

  update_window_dim_sizes <- without(
    x_shape,
    c(inserted_window_axes, x_batching_axes)
  )

  # (C4) - rank part
  expanded_si_rank <- if (index_vector_axis == si_rank + 1L) {
    si_rank + 1L
  } else {
    si_rank
  }
  expected_updates_rank <- expanded_si_rank - 1L + length(update_window_axes)
  if (updates_rank != expected_updates_rank) {
    cli_abort(c(
      "{.arg update} must have {expected_updates_rank} ax{?is/es}.",
      x = "Got {updates_rank}."
    ))
  }

  # (C4) - window sizes part
  actual_window_sizes <- updates_shape[update_window_axes]
  if (any(actual_window_sizes > update_window_dim_sizes)) {
    cli_abort(c(
      "The update window sizes must not exceed the corresponding axes of {.arg x}.",
      x = "Got {.val {actual_window_sizes}}, but the maximum allowed is {.val {update_window_dim_sizes}}."
    ))
  }

  # (C4) - scatter sizes part
  if (length(update_scatter_axes) > 0L) {
    actual_scatter_sizes <- updates_shape[update_scatter_axes]
    if (!identical(actual_scatter_sizes, update_scatter_dim_sizes)) {
      cli_abort(c(
        "The update scatter sizes must match the shape of {.arg scatter_indices} (excluding the index vector axis).",
        x = "Got {.val {actual_scatter_sizes}}, but expected {.val {update_scatter_dim_sizes}}."
      ))
    }
  }

  AbstractArray(dtype(x), x_shape)
}

# convolution(lhs, rhs, ...): compute dot products between windows of `lhs` and
# slices of `rhs`. The batch / feature / spatial axes of the input, kernel, and
# output each partition `1:rank`. The output shape places the batch size at
# `output_batch_axis`, the kernel output-feature size at `output_feature_axis`,
# and the per-spatial window counts at `output_spatial_axes`.
infer_convolution <- function(
  lhs,
  rhs,
  input_batch_axis,
  input_feature_axis,
  input_spatial_axes,
  kernel_input_feature_axis,
  kernel_output_feature_axis,
  kernel_spatial_axes,
  output_batch_axis,
  output_feature_axis,
  output_spatial_axes,
  window_strides,
  padding,
  lhs_dilation,
  rhs_dilation,
  feature_group_count,
  batch_group_count
) {
  lhs_shape <- shape(lhs)
  rhs_shape <- shape(rhs)
  rank_lhs <- naxes(lhs)
  rank_rhs <- naxes(rhs)

  # (C1)
  if (rank_lhs != rank_rhs) {
    cli_abort(c(
      "{.arg lhs} and {.arg rhs} must have the same number of axes.",
      x = "Got {rank_lhs} and {rank_rhs}."
    ))
  }
  if (rank_lhs < 2L) {
    cli_abort(c(
      "{.arg lhs} and {.arg rhs} must have at least 2 axes.",
      x = "Got {rank_lhs}."
    ))
  }
  rank <- rank_lhs
  n_spatial <- rank - 2L

  strides <- as.integer(window_strides)
  pad <- padding
  storage.mode(pad) <- "integer"
  lhs_dil <- as.integer(lhs_dilation)
  rhs_dil <- as.integer(rhs_dilation)
  fg_count <- as.integer(feature_group_count)
  bg_count <- as.integer(batch_group_count)

  # (C2)
  if (length(strides) != n_spatial) {
    cli_abort(c(
      "{.arg window_strides} must have length {n_spatial} (= number of spatial axes).",
      x = "Got {length(strides)}."
    ))
  }
  # (C3)
  if (any(strides <= 0L)) {
    cli_abort(c(
      "{.arg window_strides} must be positive.",
      x = "Got {.val {strides}}."
    ))
  }
  # (C4)
  if (nrow(pad) != n_spatial || ncol(pad) != 2L) {
    cli_abort(c(
      "{.arg padding} must have shape [{n_spatial}, 2].",
      x = "Got [{nrow(pad)}, {ncol(pad)}]."
    ))
  }
  # (C5)
  if (length(lhs_dil) != n_spatial) {
    cli_abort(c(
      "{.arg lhs_dilation} must have length {n_spatial} (= number of spatial axes).",
      x = "Got {length(lhs_dil)}."
    ))
  }
  # (C6)
  if (any(lhs_dil <= 0L)) {
    cli_abort(c(
      "{.arg lhs_dilation} must be positive.",
      x = "Got {.val {lhs_dil}}."
    ))
  }
  # (C7)
  if (length(rhs_dil) != n_spatial) {
    cli_abort(c(
      "{.arg rhs_dilation} must have length {n_spatial} (= number of spatial axes).",
      x = "Got {length(rhs_dil)}."
    ))
  }
  # (C8)
  if (any(rhs_dil <= 0L)) {
    cli_abort(c(
      "{.arg rhs_dilation} must be positive.",
      x = "Got {.val {rhs_dil}}."
    ))
  }

  # (C21)
  if (fg_count <= 0L) {
    cli_abort(c(
      "{.arg feature_group_count} must be positive.",
      x = "Got {fg_count}."
    ))
  }
  # (C22)
  if (bg_count <= 0L) {
    cli_abort(c(
      "{.arg batch_group_count} must be positive.",
      x = "Got {bg_count}."
    ))
  }
  # (C23)
  if (fg_count != 1L && bg_count != 1L) {
    cli_abort(c(
      "At least one of {.arg feature_group_count} or {.arg batch_group_count} must be 1.",
      x = "Got feature_group_count = {fg_count} and batch_group_count = {bg_count}."
    ))
  }

  # (C12)
  if (length(input_spatial_axes) != n_spatial) {
    cli_abort(c(
      "{.arg input_spatial_axes} must have length {n_spatial}.",
      x = "Got {length(input_spatial_axes)}."
    ))
  }
  # (C17)
  if (length(kernel_spatial_axes) != n_spatial) {
    cli_abort(c(
      "{.arg kernel_spatial_axes} must have length {n_spatial}.",
      x = "Got {length(kernel_spatial_axes)}."
    ))
  }
  # (C19)
  if (length(output_spatial_axes) != n_spatial) {
    cli_abort(c(
      "{.arg output_spatial_axes} must have length {n_spatial}.",
      x = "Got {length(output_spatial_axes)}."
    ))
  }

  # (C13)
  input_axes <- c(input_batch_axis, input_spatial_axes, input_feature_axis)
  if (anyDuplicated(input_axes)) {
    cli_abort(c(
      "The input batch, spatial, and feature axes must be distinct.",
      x = "Got {.val {input_axes}}."
    ))
  }
  if (axes_out_of_range(input_axes, rank)) {
    cli_abort(c(
      "The input batch, spatial, and feature axes must be between 1 and {rank}.",
      x = "Got {.val {input_axes}}."
    ))
  }

  # (C18)
  kernel_axes <- c(
    kernel_spatial_axes,
    kernel_input_feature_axis,
    kernel_output_feature_axis
  )
  if (anyDuplicated(kernel_axes)) {
    cli_abort(c(
      "The kernel spatial, input-feature, and output-feature axes must be distinct.",
      x = "Got {.val {kernel_axes}}."
    ))
  }
  if (axes_out_of_range(kernel_axes, rank)) {
    cli_abort(c(
      "The kernel spatial, input-feature, and output-feature axes must be between 1 and {rank}.",
      x = "Got {.val {kernel_axes}}."
    ))
  }

  # (C20)
  output_axes <- c(output_batch_axis, output_spatial_axes, output_feature_axis)
  if (anyDuplicated(output_axes)) {
    cli_abort(c(
      "The output batch, spatial, and feature axes must be distinct.",
      x = "Got {.val {output_axes}}."
    ))
  }
  if (axes_out_of_range(output_axes, rank)) {
    cli_abort(c(
      "The output batch, spatial, and feature axes must be between 1 and {rank}.",
      x = "Got {.val {output_axes}}."
    ))
  }

  input_batch_size <- lhs_shape[input_batch_axis]
  input_feature_size <- lhs_shape[input_feature_axis]
  kernel_input_feature_size <- rhs_shape[kernel_input_feature_axis]
  kernel_output_feature_size <- rhs_shape[kernel_output_feature_axis]

  # (C10)
  if (input_batch_size %% bg_count != 0L) {
    cli_abort(c(
      "The size of {.arg lhs} at {.arg input_batch_axis} must be divisible by {.arg batch_group_count}.",
      x = "Got size {input_batch_size} and batch_group_count {bg_count}."
    ))
  }
  # (C11)
  if (input_feature_size %% fg_count != 0L) {
    cli_abort(c(
      "The size of {.arg lhs} at {.arg input_feature_axis} must be divisible by {.arg feature_group_count}.",
      x = "Got size {input_feature_size} and feature_group_count {fg_count}."
    ))
  }
  # (C14)
  if (kernel_input_feature_size != input_feature_size %/% fg_count) {
    cli_abort(c(
      "The size of {.arg rhs} at {.arg kernel_input_feature_axis} must equal the input feature size divided by {.arg feature_group_count}.",
      x = "Got {kernel_input_feature_size}, but expected {input_feature_size %/% fg_count}."
    ))
  }
  # (C15)
  if (kernel_output_feature_size %% bg_count != 0L) {
    cli_abort(c(
      "The size of {.arg rhs} at {.arg kernel_output_feature_axis} must be divisible by {.arg batch_group_count}.",
      x = "Got size {kernel_output_feature_size} and batch_group_count {bg_count}."
    ))
  }
  # (C16)
  if (kernel_output_feature_size %% fg_count != 0L) {
    cli_abort(c(
      "The size of {.arg rhs} at {.arg kernel_output_feature_axis} must be divisible by {.arg feature_group_count}.",
      x = "Got size {kernel_output_feature_size} and feature_group_count {fg_count}."
    ))
  }

  # (C27)
  if (dtype(lhs) != dtype(rhs)) {
    cli_abort(c(
      "{.arg lhs} and {.arg rhs} must have the same dtype.",
      x = "Got {.val {as.character(dtype(lhs))}} and {.val {as.character(dtype(rhs))}}."
    ))
  }

  # (C25, C26): compute the result shape.
  result_shape <- integer(rank)
  result_shape[output_batch_axis] <- input_batch_size %/% bg_count
  result_shape[output_feature_axis] <- kernel_output_feature_size

  for (sd in seq_len(n_spatial)) {
    lhs_size <- lhs_shape[input_spatial_axes[sd]]
    rhs_size <- rhs_shape[kernel_spatial_axes[sd]]

    dilated_input <- if (lhs_size == 0L) {
      0L
    } else {
      (lhs_size - 1L) * lhs_dil[sd] + 1L
    }
    padded_input <- pad[sd, 1L] + dilated_input + pad[sd, 2L]
    dilated_window <- if (rhs_size == 0L) {
      0L
    } else {
      (rhs_size - 1L) * rhs_dil[sd] + 1L
    }
    num_windows <- if (padded_input == 0L || dilated_window > padded_input) {
      0L
    } else {
      as.integer(floor((padded_input - dilated_window) / strides[sd]) + 1L)
    }
    result_shape[output_spatial_axes[sd]] <- num_windows
  }

  AbstractArray(dtype(lhs), result_shape)
}

# dot_general(lhs, rhs, ...): batched matrix-multiply-like contraction.
# `contracting_axes` / `batching_axes` are each a list of two 1-based axis
# vectors (lhs, rhs). The output axes are the shared batch axes, followed by the
# free axes of `lhs`, followed by the free axes of `rhs`.
infer_dot_general <- function(lhs, rhs, contracting_axes, batching_axes) {
  dim_lhs <- shape(lhs)
  dim_rhs <- shape(rhs)
  rank_lhs <- naxes(lhs)
  rank_rhs <- naxes(rhs)

  lhs_contracting <- contracting_axes[[1L]]
  rhs_contracting <- contracting_axes[[2L]]
  lhs_batching <- batching_axes[[1L]]
  rhs_batching <- batching_axes[[2L]]

  # (C13)
  if (dtype(lhs) != dtype(rhs)) {
    cli_abort(c(
      "{.arg lhs} and {.arg rhs} must have the same dtype.",
      x = "Got {.val {as.character(dtype(lhs))}} and {.val {as.character(dtype(rhs))}}."
    ))
  }

  # (C1)
  if (length(lhs_batching) != length(rhs_batching)) {
    cli_abort(c(
      "{.arg batching_axes} must have equal length for {.arg lhs} and {.arg rhs}.",
      x = "Got lengths {length(lhs_batching)} and {length(rhs_batching)}."
    ))
  }
  # (C2)
  if (length(lhs_contracting) != length(rhs_contracting)) {
    cli_abort(c(
      "{.arg contracting_axes} must have equal length for {.arg lhs} and {.arg rhs}.",
      x = "Got lengths {length(lhs_contracting)} and {length(rhs_contracting)}."
    ))
  }
  # (C3)
  if (anyDuplicated(c(lhs_batching, lhs_contracting))) {
    cli_abort(c(
      "The {.arg lhs} batching and contracting axes must be distinct.",
      x = "Got {.val {c(lhs_batching, lhs_contracting)}}."
    ))
  }
  # (C4)
  if (anyDuplicated(c(rhs_batching, rhs_contracting))) {
    cli_abort(c(
      "The {.arg rhs} batching and contracting axes must be distinct.",
      x = "Got {.val {c(rhs_batching, rhs_contracting)}}."
    ))
  }
  # (C5)
  if (axes_out_of_range(lhs_batching, rank_lhs)) {
    cli_abort(c(
      "The {.arg lhs} batching axes must be between 1 and {rank_lhs}.",
      x = "Got {.val {lhs_batching}}."
    ))
  }
  # (C6)
  if (axes_out_of_range(lhs_contracting, rank_lhs)) {
    cli_abort(c(
      "The {.arg lhs} contracting axes must be between 1 and {rank_lhs}.",
      x = "Got {.val {lhs_contracting}}."
    ))
  }
  # (C7)
  if (axes_out_of_range(rhs_batching, rank_rhs)) {
    cli_abort(c(
      "The {.arg rhs} batching axes must be between 1 and {rank_rhs}.",
      x = "Got {.val {rhs_batching}}."
    ))
  }
  # (C8)
  if (axes_out_of_range(rhs_contracting, rank_rhs)) {
    cli_abort(c(
      "The {.arg rhs} contracting axes must be between 1 and {rank_rhs}.",
      x = "Got {.val {rhs_contracting}}."
    ))
  }

  dim_merge1 <- dim_lhs[lhs_contracting]
  dim_merge2 <- dim_rhs[rhs_contracting]
  dim_batch1 <- dim_lhs[lhs_batching]
  dim_batch2 <- dim_rhs[rhs_batching]

  # (C10)
  if (!identical(dim_merge1, dim_merge2)) {
    cli_abort(c(
      "The sizes of the contracting axes of {.arg lhs} and {.arg rhs} must match.",
      x = "Got {.val {dim_merge1}} for {.arg lhs} and {.val {dim_merge2}} for {.arg rhs}."
    ))
  }
  # (C9)
  if (!identical(dim_batch1, dim_batch2)) {
    cli_abort(c(
      "The sizes of the batching axes of {.arg lhs} and {.arg rhs} must match.",
      x = "Got {.val {dim_batch1}} for {.arg lhs} and {.val {dim_batch2}} for {.arg rhs}."
    ))
  }

  dim_lhs_remaining <- without(dim_lhs, c(lhs_contracting, lhs_batching))
  dim_rhs_remaining <- without(dim_rhs, c(rhs_contracting, rhs_batching))
  # (C12)
  out_dim <- c(dim_batch1, dim_lhs_remaining, dim_rhs_remaining)

  AbstractArray(dtype(lhs), out_dim)
}

# --- Reduction / cumulative / arg-extreme (native, moved from primitives.R) ---

infer_reduce <- function(x, axes, drop) {
  old_shape <- shape(x)
  if (drop) {
    new_shape <- old_shape[-axes]
  } else {
    new_shape <- old_shape
    new_shape[axes] <- 1L
  }
  list(AbstractArray(
    dtype = dtype(x),
    shape = Shape(new_shape),
    ambiguous = x$ambiguous
  ))
}

infer_reduce_boolean <- function(x, axes, drop) {
  old_shape <- shape(x)
  if (drop) {
    new_shape <- old_shape[-axes]
  } else {
    new_shape <- old_shape
    new_shape[axes] <- 1L
  }
  list(AbstractArray(
    dtype = "bool",
    shape = Shape(new_shape),
    ambiguous = FALSE
  ))
}

infer_cum <- function(x, axis) {
  rank <- length(shape(x))
  if (rank == 0L) {
    cli_abort("cumulative ops require at least a 1-dimensional {.arg x}, but it is a scalar")
  }
  if (!checkmate::test_integerish(axis, lower = 1, upper = rank, len = 1L)) {
    cli_abort("{.arg axis} must be a single integer in 1:{rank}, but is {.val {axis}}")
  }
  list(AbstractArray(
    dtype = dtype(x),
    shape = Shape(shape(x)),
    ambiguous = x$ambiguous
  ))
}

infer_cum_extreme <- function(x, axis) {
  rank <- length(shape(x))
  if (rank == 0L) {
    cli_abort("cumulative ops require at least a 1-dimensional {.arg x}, but it is a scalar")
  }
  if (!checkmate::test_integerish(axis, lower = 1, upper = rank, len = 1L)) {
    cli_abort("{.arg axis} must be a single integer in 1:{rank}, but is {.val {axis}}")
  }
  list(
    AbstractArray(
      dtype = dtype(x),
      shape = Shape(shape(x)),
      ambiguous = x$ambiguous
    ),
    AbstractArray(
      dtype = "i32",
      shape = Shape(shape(x)),
      ambiguous = FALSE
    )
  )
}

infer_fn_arg_extreme <- function(x, axis, drop) {
  shp <- shape(x)
  if (axis > length(shp)) {
    cli_abort(c(
      "{.arg axis} is out of bounds.",
      x = "`x` has {length(shp)} axes, got {.arg axis} = {axis}."
    ))
  }
  # The reduction lowering uses `init_v = +/-Inf` and `init_i = 0`. Reducing
  # along a size-0 axis would silently emit those sentinels (i.e. index 1)
  # rather than failing. Argmax/argmin of an empty axis is undefined, so
  # reject it here at trace time.
  if (shp[axis] == 0L) {
    cli_abort(c(
      "argmax/argmin is undefined for an empty axis.",
      x = "`x` has shape {xlamisc::shapevec_repr(shp)}; {.arg axis} = {axis} has size 0."
    ))
  }
  if (drop) {
    new_shape <- shp[-axis]
  } else {
    new_shape <- shp
    new_shape[axis] <- 1L
  }
  list(AbstractArray(
    dtype = "i32",
    shape = Shape(new_shape),
    ambiguous = FALSE
  ))
}
