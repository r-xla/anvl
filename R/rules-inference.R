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
  if (!inherits(dtype(x), classes)) {
    cli_abort(c(
      "{.arg {arg}} must have dtype {.or {classes}}.",
      x = "Got {.val {as.character(dtype(x))}}."
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
  if (inherits(dtype(x), "BooleanType")) {
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
  if (!inherits(dt, c("IntegerType", "UIntegerType", "FloatType"))) {
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
  if (!inherits(out_dtype, c("IntegerType", "UIntegerType", "FloatType"))) {
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
