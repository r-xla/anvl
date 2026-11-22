#' @include primitives.R

# Type inference was already done at trace time. Rules that declare an
# `output_types` parameter receive the known result types from the lowering
# driver (see stablehlo()) and forward them to their hlo_* builder, letting
# stablehlo skip re-inference. Composite and region rules (reduce, cumulative
# ops, linalg decompositions, ...) omit the parameter and keep normal inference.

index_dtype_of <- function(output_types, i) {
  as.character(output_types[[i]]$type$dtype)
}

region_input <- function(dtype, shape = integer()) {
  func <- stablehlo::.current_func()
  vt <- stablehlo::ValueType(dtype, shape = shape)
  id <- stablehlo::ValueId()
  func$inputs <- stablehlo::FuncInputs(c(
    func$inputs,
    list(stablehlo::FuncInput(id, vt))
  ))
  stablehlo::FuncValue(id, vt, func)
}

# Indices shifted from 1-based to 0-based, and back. An unsigned index is first
# converted to a signed data type wide enough to hold it, so that index 0
# becomes -1 (out of bounds below) instead of wrapping around to the largest
# value.
hlo_zero_based_indices <- function(indices) {
  dt <- dtype(indices)
  if (is_dtype_uint(dt)) {
    indices <- hlo_convert(indices, as_dtype(paste0("i", min(2L * dtype_width(dt), 64L))))
  }
  hlo_subtract(indices, hlo_ones_like(indices))
}

hlo_one_based_indices <- function(indices) {
  hlo_add(indices, hlo_ones_like(indices))
}

hlo_ones_like <- function(x) {
  hlo_tensor(1L, dtype = dtype(x), shape = shape(x$value_type), func = x$func)
}

# A reduction's result, reshaped to keep the reduced R `axes` of `x` at size 1
# unless `drop`.
hlo_reduced_output <- function(out, x, axes, drop) {
  if (drop) {
    return(list(out))
  }
  list(hlo_reshape(out, xla_shape(replace(r_shape(x), axes, 1L))))
}

# An operand of an elementwise rule at the shape of the output: a scalar is
# broadcast to it, any other operand already has it.
hlo_broadcast_scalar <- function(x, output_types) {
  out_shape <- shape(output_types[[1L]])
  if (naxes(x) || !length(out_shape)) {
    return(x)
  }
  hlo_broadcast_in_dim(x, integer(), out_shape)
}

# The rule of an elementwise primitive, whose operands are scalars or have the
# shape of the output. `hlo_fn` receives the operands, each broadcast to that
# shape, and `output_types`. It stays unevaluated until the rule first runs:
# `elementwise_rule(hlo_add)` names an import of stablehlo, and
# `devtools::document()` loads the package before it writes the imports.
elementwise_rule <- function(hlo_fn) {
  function(..., output_types) {
    operands <- lapply(list(...), hlo_broadcast_scalar, output_types = output_types)
    list(do.call(hlo_fn, c(unname(operands), list(output_types = output_types))))
  }
}

prim_fill[["stablehlo"]] <- function(value, shape, dtype) {
  list(hlo_tensor(value, shape = xla_shape(shape), dtype = dtype))
}

prim_add[["stablehlo"]] <- elementwise_rule(hlo_add)

prim_mul[["stablehlo"]] <- elementwise_rule(hlo_multiply)

prim_sub[["stablehlo"]] <- elementwise_rule(hlo_subtract)

prim_negate[["stablehlo"]] <- function(x, output_types) {
  list(hlo_negate(x, output_types = output_types))
}

prim_div[["stablehlo"]] <- elementwise_rule(hlo_divide)

prim_pow[["stablehlo"]] <- elementwise_rule(hlo_power)

prim_broadcast_in_axes[["stablehlo"]] <- function(x, shape, broadcast_axes, output_types) {
  dims <- xla_broadcast_dims(broadcast_axes, length(shape))
  list(hlo_broadcast_in_dim(x, dims, xla_shape(shape), output_types = output_types))
}

# The result of `prim_dot_general()` in R order is the free axes of `lhs`, then
# those of `rhs`, then the batch axes (in the order `batching_axes` names
# them); its XLA order is the reverse: the batch dimensions in reverse, then the
# free dimensions of `rhs`, then those of `lhs`, each in XLA order. XLA's
# `dot_general` puts its batch dimensions first, in the order they are named,
# and then each operand's free dimensions in XLA order, so with the operands
# swapped and the batch dimensions named in reverse it produces exactly the
# result.
prim_dot_general[["stablehlo"]] <- function(lhs, rhs, contracting_axes, batching_axes, precision, output_types) {
  n_lhs <- naxes(lhs)
  n_rhs <- naxes(rhs)
  contracting <- list(xla_dims(contracting_axes[[2L]], n_rhs), xla_dims(contracting_axes[[1L]], n_lhs))
  batching <- list(xla_dims(rev(batching_axes[[2L]]), n_rhs), xla_dims(rev(batching_axes[[1L]]), n_lhs))
  list(hlo_dot_general(
    rhs,
    lhs,
    contracting,
    batching,
    precision_config = toupper(precision),
    output_types = output_types
  ))
}

prim_scaled_dot_product_attention[["stablehlo"]] <- function(x, key, value, scale, causal, compute_dtype) {
  query_shape <- r_shape(x)
  value_shape <- r_shape(value)
  rank <- length(query_shape)
  dt <- as.character(dtype(x))
  compute_dtype <- compute_dtype %||% dt
  feature_size <- query_shape[[rank]]
  if (
    !identical(current_platform(), "cuda") ||
      !compute_dtype %in% c("bf16", "f16") ||
      !all(query_shape > 0L) ||
      !all(value_shape > 0L) ||
      feature_size != value_shape[[rank]] ||
      feature_size > 128L ||
      feature_size %% 8L != 0L ||
      (causal && query_shape[[rank - 1L]] != value_shape[[rank - 1L]])
  ) {
    return(NULL)
  }

  # cuDNN expects leading batch axes followed by positions and features.
  reverse_axes <- function(x) hlo_transpose(x, rev(seq_len(naxes(x)) - 1L))
  x <- reverse_axes(hlo_convert(x, compute_dtype))
  key <- reverse_axes(hlo_convert(key, compute_dtype))
  value <- reverse_axes(hlo_convert(value, compute_dtype))
  scale <- scale %||% (1 / sqrt(feature_size))
  out <- hlo_cudnn_attention(x, key, value, scale, causal)
  list(reverse_axes(hlo_convert(out, dt)))
}

# XLA's cuDNN FMHA custom call expects [batch, heads, positions, features]
# in row-major layout, plus a second result for compiler-sized workspace.
hlo_cudnn_attention <- function(x, key, value, scale, causal) {
  reshape_for_cudnn <- function(x) {
    shp <- shape(x)
    rank <- length(shp)
    if (rank == 4L) {
      return(x)
    }
    heads <- if (rank >= 3L) shp[[rank - 2L]] else 1L
    batch <- prod(shp[seq_len(max(0L, rank - 3L))])
    hlo_reshape(x, c(batch, heads, tail(shp, 2L)))
  }
  query_shape <- shape(x)
  x <- reshape_for_cudnn(x)
  key <- reshape_for_cudnn(key)
  value <- reshape_for_cudnn(value)
  bnth <- shape(x)
  value_bnth <- shape(value)
  layout <- 3:0
  # Protobuf repeated fields require JSON arrays, including singleton axes.
  dot_axes <- function(rhs_contracting) {
    list(
      lhs_contracting_dimensions = list("3"),
      rhs_contracting_dimensions = list(rhs_contracting),
      lhs_batch_dimensions = c("0", "1"),
      rhs_batch_dimensions = c("0", "1")
    )
  }
  config <- jsonlite::toJSON(
    list(
      cudnn_fmha_backend_config = list(
        algorithm = list(
          algo_id = "0",
          math_type = "TENSOR_OP_MATH",
          tuning_knobs = list("17" = "1", "24" = "0"),
          is_cudnn_frontend = TRUE,
          workspace_size = "0"
        ),
        fmha_scale = scale,
        intermediate_tensor_shape = list(
          element_type = toupper(as.character(dtype(x))),
          dimensions = as.character(c(bnth[1:3], value_bnth[[3L]])),
          layout = list(minor_to_major = as.character(layout))
        ),
        is_flash_attention = TRUE,
        mask_type = if (causal) "CAUSAL" else "NO_MASK",
        max_seg_per_batch = 1L,
        bmm1_dot_dimension_numbers = dot_axes("3"),
        bmm2_dot_dimension_numbers = dot_axes("2")
      )
    ),
    auto_unbox = TRUE,
    digits = NA
  )
  out <- hlo_custom_call(
    x,
    key,
    value,
    call_target_name = "__cudnn$fmhaSoftmax",
    api_version = 2L,
    has_side_effect = FALSE,
    backend_config = as.character(config),
    output_types = list(vt(dtype(x), bnth), vt("ui8", 0L)),
    operand_layouts = rep(list(layout), 3L),
    result_layouts = list(layout, 0L)
  )[[1L]]
  if (length(query_shape) == 4L) out else hlo_reshape(out, query_shape)
}

prim_transpose[["stablehlo"]] <- function(x, perm, output_types) {
  list(hlo_transpose(x, xla_perm(perm), output_types = output_types))
}

# A column-major reshape, like base R's `dim<-`, is a row-major reshape of the
# reversed shapes, which is what `hlo_reshape()` does.
prim_reshape[["stablehlo"]] <- function(x, shape, output_types) {
  list(hlo_reshape(x, xla_shape(shape), output_types = output_types))
}

prim_concatenate[["stablehlo"]] <- function(..., axis, output_types) {
  n <- naxes(..1)
  list(hlo_concatenate(..., dimension = xla_dims(axis, n), output_types = output_types))
}

prim_static_slice[["stablehlo"]] <- function(x, start_indices, end_indices, strides, output_types) {
  # we use 1:n, which includes n, but this translates to 0:n in stablehlo
  list(hlo_slice(
    x,
    xla_per_axis(start_indices - 1L),
    xla_per_axis(end_indices),
    xla_per_axis(strides),
    output_types = output_types
  ))
}

prim_dynamic_slice[["stablehlo"]] <- function(x, ..., slice_sizes, output_types) {
  start_indices_0based <- xla_per_axis(lapply(list(...), hlo_zero_based_indices))
  list(rlang::exec(
    hlo_dynamic_slice,
    x,
    !!!start_indices_0based,
    slice_sizes = xla_per_axis(slice_sizes),
    output_types = output_types
  ))
}

prim_dynamic_update_slice[["stablehlo"]] <- function(x, update, ..., output_types) {
  start_indices <- list(...)
  if (!length(start_indices)) {
    return(list(update))
  }
  start_indices_0based <- xla_per_axis(lapply(start_indices, hlo_zero_based_indices))
  list(rlang::exec(
    hlo_dynamic_update_slice,
    x,
    update,
    !!!start_indices_0based,
    output_types = output_types
  ))
}


.stablehlo_apply_reduce <- function(reducer, x, init, axes, drop) {
  local_func("")
  dt <- as.character(x$value_type$type$dtype)
  f <- hlo_return(reducer(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  out <- hlo_reduce(list(x), list(init(x)), xla_dims(axes, naxes(x)), f)
  hlo_reduced_output(out, x, axes, drop)
}

prim_sum[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    hlo_scalar(0L, dtype = dtype(x), func = x$func)
  }
  .stablehlo_apply_reduce(hlo_add, x, init, axes, drop)
}

prim_prod[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    hlo_scalar(1L, dtype = dtype(x), func = x$func)
  }
  .stablehlo_apply_reduce(hlo_multiply, x, init, axes, drop)
}


prim_max[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_minval(dtype(x), "cpu"))
  }
  .stablehlo_apply_reduce(hlo_maximum, x, init, axes, drop)
}

prim_min[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_maxval(dtype(x), "cpu"))
  }
  .stablehlo_apply_reduce(hlo_minimum, x, init, axes, drop)
}

prim_any[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    hlo_scalar(FALSE)
  }
  .stablehlo_apply_reduce(hlo_or, x, init, axes, drop)
}

prim_all[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    hlo_scalar(TRUE)
  }
  .stablehlo_apply_reduce(hlo_and, x, init, axes, drop)
}

# A cumulative reduction along `axis` of the array `x`, written as a
# reduce_window whose window spans the whole axis and is padded on the low side,
# as JAX does; XLA optimizes this pattern.
.hlo_cum_reduce_window <- function(x, inputs, init_values, body, axis) {
  shp <- shape(x$value_type)
  n_axes <- length(shp)
  # the 1-based position of R axis `axis` in the XLA shape
  pos <- xla_dims(axis, n_axes) + 1L
  # reduce_window rejects a window of size 0, and there is nothing to accumulate.
  if (shp[[pos]] == 0L) {
    return(inputs)
  }
  window_dimensions <- replace(rep(1L, n_axes), pos, shp[[pos]])
  ones <- rep(1L, n_axes)
  padding <- matrix(0L, nrow = n_axes, ncol = 2L)
  padding[pos, 1L] <- shp[[pos]] - 1L
  hlo_reduce_window(
    inputs = inputs,
    init_values = init_values,
    window_dimensions = window_dimensions,
    window_strides = ones,
    base_dilations = ones,
    window_dilations = ones,
    padding = padding,
    body = body
  )
}

.stablehlo_apply_cum <- function(reducer, x, init, axis) {
  local_func("")
  dt <- as.character(x$value_type$type$dtype)
  body <- hlo_return(reducer(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  list(.hlo_cum_reduce_window(x, x, init(x), body, axis))
}

prim_cumsum[["stablehlo"]] <- function(x, axis) {
  init <- function(x) {
    hlo_scalar(0L, dtype = dtype(x), func = x$func)
  }
  .stablehlo_apply_cum(hlo_add, x, init, axis)
}

prim_cumprod[["stablehlo"]] <- function(x, axis) {
  init <- function(x) {
    hlo_scalar(1L, dtype = dtype(x), func = x$func)
  }
  .stablehlo_apply_cum(hlo_multiply, x, init, axis)
}

# Here we also return the indices to simplify pullback rule
.stablehlo_apply_cum_extreme <- function(x, axis, is_max, index_dtype) {
  v_dtype <- as.character(x$value_type$type$dtype)
  i_dtype <- as.character(index_dtype)
  iota <- hlo_iota(
    iota_dimension = xla_dims(axis, naxes(x)),
    dtype = i_dtype,
    shape = shape(x$value_type)
  )

  init_v_fn <- if (is_max) nv_minval else nv_maxval
  init_v <- hlo_scalar(init_v_fn(v_dtype, "cpu"))
  init_i <- hlo_scalar(0L, dtype = i_dtype, func = x$func)

  cmp <- if (is_max) prim_gt else prim_lt
  is_float <- is_dtype_float(x$value_type$type$dtype)
  # Last-occurrence tiebreak: on equal values pick the larger index. XLA's
  # reduce_window is associative but applied in an implementation-defined order
  # (and is not commutative), so we cannot assume the rhs holds the larger
  # index -- on CUDA the accumulator index can exceed the incoming one. We
  # therefore break ties explicitly on `li > ri` (same fix as
  # .stablehlo_arg_extreme, mirrored for last-occurrence).
  #
  # For float inputs the reducer is NaN-propagating: if either side is NaN,
  # the result is NaN. Without this, XLA's comparison-based max/min gives
  # `max(state, NaN) = NaN` then `max(NaN, next) = next`, which makes the
  # running extreme restart after every NaN — useless for users.
  lhs_wins_fn <- function(lv, li, rv, ri) {
    nv_or(cmp(lv, rv), nv_and(prim_eq(lv, rv), prim_gt(li, ri)))
  }
  reducer <- if (is_float) {
    function(lv, li, rv, ri) {
      either_nan <- prim_or(prim_ne(lv, lv), prim_ne(rv, rv))
      lhs_wins <- lhs_wins_fn(lv, li, rv, ri)
      out_v <- nv_ifelse(either_nan, NaN, nv_ifelse(lhs_wins, lv, rv))
      out_i <- nv_ifelse(lhs_wins, li, ri)
      list(out_v, out_i)
    }
  } else {
    function(lv, li, rv, ri) {
      lhs_wins <- lhs_wins_fn(lv, li, rv, ri)
      list(nv_ifelse(lhs_wins, lv, rv), nv_ifelse(lhs_wins, li, ri))
    }
  }
  body <- .r_reducer_to_hlo_func(reducer, .value_index_avals(v_dtype, i_dtype))

  out <- .hlo_cum_reduce_window(x, list(x, iota), list(init_v, init_i), body, axis)
  list(out[[1L]], hlo_one_based_indices(out[[2L]]))
}

# The index outputs follow the default integer data type, which the trace has
# already materialized at: it is read off the inferred output types rather than
# from `default_int()`, which lowering runs too late to consult.
prim_cummax[["stablehlo"]] <- function(x, axis, output_types) {
  .stablehlo_apply_cum_extreme(x, axis, is_max = TRUE, index_dtype = index_dtype_of(output_types, 2L))
}

prim_cummin[["stablehlo"]] <- function(x, axis, output_types) {
  .stablehlo_apply_cum_extreme(x, axis, is_max = FALSE, index_dtype = index_dtype_of(output_types, 2L))
}

prim_reduce[["stablehlo"]] <- function(x, init, ..., axes, drop, reducer, n_captures) {
  red_func <- stablehlo(reducer, id = "", captured = list(...))[[1L]]
  out <- hlo_reduce(
    inputs = list(x),
    init_values = list(init),
    dimensions = xla_dims(axes, naxes(x)),
    body = red_func
  )
  hlo_reduced_output(out, x, axes, drop)
}

.r_reducer_to_hlo_func <- function(fn, dummy_args) {
  graph <- trace_fn(fn, dummy_args, desc = local_descriptor())
  stablehlo(graph, id = "")[[1L]]
}

# The scalar arguments of a reducer over (value, index) pairs: the left pair,
# then the right one.
.value_index_avals <- function(v_dtype, i_dtype) {
  list(
    lv = nv_aval(v_dtype, integer()),
    li = nv_aval(i_dtype, integer()),
    rv = nv_aval(v_dtype, integer()),
    ri = nv_aval(i_dtype, integer())
  )
}

.stablehlo_arg_extreme <- function(x, axis, drop, direction, init_v_fn, index_dtype) {
  shp <- shape(x$value_type)
  reduced <- xla_dims(axis, length(shp))
  v_dtype <- x$value_type$type$dtype
  i_dtype <- as.character(index_dtype)
  iota <- hlo_iota(iota_dimension = reduced, dtype = i_dtype, shape = shp)
  init_v <- hlo_scalar(init_v_fn(v_dtype, "cpu"))
  init_i <- hlo_scalar(0L, dtype = i_dtype, func = x$func)

  # Reductor: pick lhs unless rhs is strictly better. On a tie, pick the
  # smaller index. XLA's reduce is associative but may be applied in any order
  # (and is not commutative), so we cannot assume li < ri -- on CUDA the
  # accumulator index can exceed the incoming index. We therefore break ties
  # explicitly on `ri < li` rather than relying on argument order.
  cmp <- if (direction == "GT") prim_gt else prim_lt
  reducer <- function(lv, li, rv, ri) {
    rhs_better <- nv_or(cmp(rv, lv), nv_and(prim_eq(rv, lv), prim_lt(ri, li)))
    list(nv_ifelse(rhs_better, rv, lv), nv_ifelse(rhs_better, ri, li))
  }
  body <- .r_reducer_to_hlo_func(reducer, .value_index_avals(v_dtype, i_dtype))

  out <- hlo_reduce(
    inputs = list(x, iota),
    init_values = list(init_v, init_i),
    dimensions = reduced,
    body = body
  )
  hlo_reduced_output(hlo_one_based_indices(out[[2L]]), x, axis, drop)
}

prim_which_max[["stablehlo"]] <- function(x, axis, drop, output_types) {
  .stablehlo_arg_extreme(
    x,
    axis,
    drop,
    direction = "GT",
    init_v_fn = nv_minval,
    index_dtype = index_dtype_of(output_types, 1L)
  )
}

prim_which_min[["stablehlo"]] <- function(x, axis, drop, output_types) {
  .stablehlo_arg_extreme(
    x,
    axis,
    drop,
    direction = "LT",
    init_v_fn = nv_maxval,
    index_dtype = index_dtype_of(output_types, 1L)
  )
}

# comparison jit rules ----------------------------------------------------------

.compare_type_for <- function(x) {
  dt <- x$value_type$type$dtype
  if (is_dtype_float(dt)) {
    "FLOAT"
  } else if (is_dtype_int(dt)) {
    "SIGNED"
  } else if (is_dtype_uint(dt) || is_dtype_bool(dt)) {
    "UNSIGNED"
  } else {
    cli_abort("Unsupported dtype for compare")
  }
}

.stablehlo_compare_bin <- function(direction) {
  elementwise_rule(function(lhs, rhs, output_types) {
    hlo_compare(
      lhs,
      rhs,
      comparison_direction = direction,
      compare_type = .compare_type_for(lhs),
      output_types = output_types
    )
  })
}

prim_eq[["stablehlo"]] <- .stablehlo_compare_bin("EQ")
prim_ne[["stablehlo"]] <- .stablehlo_compare_bin("NE")
prim_gt[["stablehlo"]] <- .stablehlo_compare_bin("GT")
prim_ge[["stablehlo"]] <- .stablehlo_compare_bin("GE")
prim_lt[["stablehlo"]] <- .stablehlo_compare_bin("LT")
prim_le[["stablehlo"]] <- .stablehlo_compare_bin("LE")


# binary simple math jit rules ---------------------------------------------------

prim_pmax[["stablehlo"]] <- elementwise_rule(hlo_maximum)

prim_pmin[["stablehlo"]] <- elementwise_rule(hlo_minimum)

prim_remainder[["stablehlo"]] <- elementwise_rule(hlo_remainder)

prim_and[["stablehlo"]] <- elementwise_rule(hlo_and)

prim_not[["stablehlo"]] <- function(x, output_types) {
  list(hlo_not(x, output_types = output_types))
}

prim_or[["stablehlo"]] <- elementwise_rule(hlo_or)

prim_xor[["stablehlo"]] <- elementwise_rule(hlo_xor)

prim_shift_left[["stablehlo"]] <- elementwise_rule(hlo_shift_left)

prim_shift_right_logical[["stablehlo"]] <- elementwise_rule(hlo_shift_right_logical)

prim_shift_right_arithmetic[["stablehlo"]] <- elementwise_rule(hlo_shift_right_arithmetic)

prim_atan2[["stablehlo"]] <- elementwise_rule(hlo_atan2)

# anvl's leading lane axis is XLA's trailing dimension, the one stablehlo's
# bitcast_convert consumes or produces.
prim_bitcast_convert[["stablehlo"]] <- function(x, dtype, output_types) {
  list(hlo_bitcast_convert(x, dtype, output_types = output_types))
}

# unary simple math jit rules ---------------------------------------------------

prim_abs[["stablehlo"]] <- function(x, output_types) {
  list(hlo_abs(x, output_types = output_types))
}

prim_sqrt[["stablehlo"]] <- function(x, output_types) {
  list(hlo_sqrt(x, output_types = output_types))
}

prim_rsqrt[["stablehlo"]] <- function(x, output_types) {
  list(hlo_rsqrt(x, output_types = output_types))
}

prim_log[["stablehlo"]] <- function(x, output_types) {
  list(hlo_log(x, output_types = output_types))
}

prim_tanh[["stablehlo"]] <- function(x, output_types) {
  list(hlo_tanh(x, output_types = output_types))
}

prim_tan[["stablehlo"]] <- function(x, output_types) {
  list(hlo_tan(x, output_types = output_types))
}

prim_sin[["stablehlo"]] <- function(x, output_types) {
  list(hlo_sine(x, output_types = output_types))
}

prim_cos[["stablehlo"]] <- function(x, output_types) {
  list(hlo_cosine(x, output_types = output_types))
}

prim_floor[["stablehlo"]] <- function(x, output_types) {
  list(hlo_floor(x, output_types = output_types))
}

prim_ceiling[["stablehlo"]] <- function(x, output_types) {
  list(hlo_ceil(x, output_types = output_types))
}

prim_sign[["stablehlo"]] <- function(x, output_types) {
  list(hlo_sign(x, output_types = output_types))
}

prim_exp[["stablehlo"]] <- function(x, output_types) {
  list(hlo_exponential(x, output_types = output_types))
}

prim_expm1[["stablehlo"]] <- function(x, output_types) {
  list(hlo_exponential_minus_one(x, output_types = output_types))
}

prim_log1p[["stablehlo"]] <- function(x, output_types) {
  list(hlo_log_plus_one(x, output_types = output_types))
}

prim_cbrt[["stablehlo"]] <- function(x, output_types) {
  list(hlo_cbrt(x, output_types = output_types))
}

prim_plogis[["stablehlo"]] <- function(x, output_types) {
  list(hlo_logistic(x, output_types = output_types))
}

prim_acos[["stablehlo"]] <- function(x, output_types) {
  list(hlo_acos(x, output_types = output_types))
}

prim_acosh[["stablehlo"]] <- function(x, output_types) {
  list(hlo_acosh(x, output_types = output_types))
}

prim_asin[["stablehlo"]] <- function(x, output_types) {
  list(hlo_asin(x, output_types = output_types))
}

prim_asinh[["stablehlo"]] <- function(x, output_types) {
  list(hlo_asinh(x, output_types = output_types))
}

prim_atan[["stablehlo"]] <- function(x, output_types) {
  list(hlo_atan(x, output_types = output_types))
}

prim_atanh[["stablehlo"]] <- function(x, output_types) {
  list(hlo_atanh(x, output_types = output_types))
}

prim_cosh[["stablehlo"]] <- function(x, output_types) {
  list(hlo_cosh(x, output_types = output_types))
}

prim_sinh[["stablehlo"]] <- function(x, output_types) {
  list(hlo_sinh(x, output_types = output_types))
}

prim_digamma[["stablehlo"]] <- function(x, output_types) {
  list(hlo_digamma(x, output_types = output_types))
}

prim_lgamma[["stablehlo"]] <- function(x, output_types) {
  list(hlo_lgamma(x, output_types = output_types))
}

prim_psigamma[["stablehlo"]] <- elementwise_rule(function(x, deriv, output_types) {
  hlo_polygamma(deriv, x, output_types = output_types)
})

prim_erf[["stablehlo"]] <- function(x, output_types) {
  list(hlo_erf(x, output_types = output_types))
}

prim_erf_inv[["stablehlo"]] <- function(x, output_types) {
  list(hlo_erf_inv(x, output_types = output_types))
}

prim_erfc[["stablehlo"]] <- function(x, output_types) {
  list(hlo_erfc(x, output_types = output_types))
}

prim_is_finite[["stablehlo"]] <- function(x, output_types) {
  list(hlo_is_finite(x, output_types = output_types))
}

prim_popcnt[["stablehlo"]] <- function(x, output_types) {
  list(hlo_popcnt(x, output_types = output_types))
}

# stablehlo's clamp takes scalar bounds as they are, but not a scalar operand
# between bounds that are arrays.
prim_clamp[["stablehlo"]] <- function(x, min, max, output_types) {
  list(hlo_clamp(min, hlo_broadcast_scalar(x, output_types), max, output_types = output_types))
}

prim_rev[["stablehlo"]] <- function(x, axes, output_types) {
  list(hlo_reverse(x, xla_dims(axes, naxes(x)), output_types = output_types))
}

prim_iota[["stablehlo"]] <- function(axis, shape, dtype, start) {
  out <- hlo_iota(iota_dimension = xla_dims(axis, length(shape)), dtype = dtype, shape = xla_shape(shape))
  if (start != 0L) {
    offset <- hlo_broadcast_in_dim(
      hlo_scalar(start, dtype = dtype, func = out$func),
      integer(0L),
      xla_shape(shape)
    )
    out <- hlo_add(out, offset)
  }
  list(out)
}

prim_pad[["stablehlo"]] <- function(
  x,
  padding_value,
  edge_padding_low,
  edge_padding_high,
  interior_padding,
  output_types
) {
  list(hlo_pad(
    x,
    padding_value,
    xla_per_axis(edge_padding_low),
    xla_per_axis(edge_padding_high),
    xla_per_axis(interior_padding),
    output_types = output_types
  ))
}

prim_round[["stablehlo"]] <- function(x, method) {
  switch(
    method,
    afz = list(hlo_round_nearest_afz(x)),
    nearest_even = list(hlo_round_nearest_even(x)),
    cli_abort("invalid method: {method}")
  )
}

prim_convert[["stablehlo"]] <- function(x, dtype, output_types) {
  list(hlo_convert(x, dtype, output_types = output_types))
}


# stablehlo's select takes a scalar `test` as it is, but not scalar branches
# for a `test` that is an array.
prim_ifelse[["stablehlo"]] <- function(test, yes, no, output_types) {
  yes <- hlo_broadcast_scalar(yes, output_types)
  no <- hlo_broadcast_scalar(no, output_types)
  list(hlo_select(test, yes, no, output_types = output_types))
}

# RNG jit rules --------------------------------------------------------

# A random state is a ThreeFry-2x32 key, `ui32[2]` (the words of JAX's key data), and a
# counter, `ui64`. ?prim_sample_bits says which values a sampling call draws.
# Each sampling call and each fork is a call of a function built once per
# lowering (and draw shape and data type), so that many sampling calls lower
# and compile quickly.
prim_sample_bits[["stablehlo"]] <- function(key, counter, shape, dtype) {
  shape <- as.integer(shape)
  dtype <- as.character(as_dtype(dtype))
  rng_state_call(paste0("sample_bits_", dtype, "_", shape_id(shape)), key, counter, function(fresh_key) {
    list(threefry_draw(fresh_key, shape, dtype))
  })
}

prim_fork_rng_state[["stablehlo"]] <- function(key, counter) {
  rng_state_call("fork_rng_state", key, counter, function(fresh_key) {
    words <- lapply(fresh_key, hlo_reshape, 1L)
    list(
      hlo_concatenate(words[[1L]], words[[2L]], dimension = 0L),
      hlo_scalar(0L, dtype = "ui64", func = fresh_key[[1L]]$func)
    )
  })
}

# A call of the function `id` of the lowering, built on first use, that splits
# the random state with the key `key` at the counter `counter` and returns what
# `body()` computes from the key ThreeFry_key(counter) (two `ui32` scalars). It
# encrypts the counter with its high word as the first input word and its low
# word as the second: for a counter c below 2^32 the key is JAX's
# `fold_in(key, c)`, and for every c the c-th key of JAX's `split(key, c + 1)`.
rng_state_call <- function(id, key, counter, body) {
  callee <- lowering_func(id, function(id) {
    func <- local_func(id)
    key <- stablehlo::hlo_input("key", "ui32", shape = 2L, func = func)
    counter <- stablehlo::hlo_input("counter", "ui64", func = func)
    key_words <- lapply(0:1, function(i) hlo_reshape(hlo_slice(key, i, i + 1L, 1L), integer()))
    # each word is cleared of the other first, so that the conversion is exact
    by32 <- hlo_scalar(32L, dtype = "ui64", func = func)
    low <- hlo_shift_right_logical(hlo_shift_left(counter, by32), by32)
    high <- hlo_shift_right_logical(counter, by32)
    counter_words <- list(hlo_convert(high, "ui32"), hlo_convert(low, "ui32"))
    do.call(stablehlo::hlo_return, c(body(threefry2x32(key_words, counter_words)), list(func = func)))
  })
  stablehlo::hlo_call(callee, key, counter, simplify = FALSE)
}

# `shape` as part of a function name.
shape_id <- function(shape) {
  if (length(shape)) paste(shape, collapse = "x") else "scalar"
}

# The values of R shape `shape` and unsigned data type `dtype` a draw from the
# key `key` (two `ui32` scalars) gives: the value at the column-major index t is
# made from the block that encrypts the input words (0, t) alone, so that it
# does not depend on how many values the draw has. This is how JAX's
# `jax.random.bits()` draws by default (`jax_threefry_partitionable`), except
# that JAX fills a shape in row-major order. A `ui64` value is the
# block's first word shifted into the high half and its second word in the low
# half, and a narrower value the xor of the two words, truncated to its low
# bits.
threefry_draw <- function(key, shape, dtype) {
  func <- key[[1L]]$func
  n <- prod(shape)
  if (n > .Machine$integer.max) {
    limit <- .Machine$integer.max
    cli_abort(c("Cannot draw more than {limit} values at once.", i = "Draw fewer values at once."))
  }
  n <- as.integer(n)
  zeros <- hlo_broadcast_in_dim(hlo_scalar(0L, dtype = "ui32", func = func), integer(), n)
  index <- hlo_iota(iota_dimension = 0L, dtype = "ui32", shape = n, func = func)
  words <- threefry2x32(lapply(key, hlo_broadcast_in_dim, integer(), n), list(zeros, index))
  if (dtype == "ui64") {
    by32 <- hlo_broadcast_in_dim(hlo_scalar(32L, dtype = "ui64", func = func), integer(), n)
    high <- hlo_shift_left(hlo_convert(words[[1L]], "ui64"), by32)
    return(hlo_reshape(hlo_or(high, hlo_convert(words[[2L]], "ui64")), xla_shape(shape)))
  }
  values <- hlo_xor(words[[1L]], words[[2L]])
  if (dtype != "ui32") {
    # keep the low bits, so that the conversion is exact
    mask <- hlo_broadcast_in_dim(
      hlo_scalar(as.integer(2^dtype_width(as_dtype(dtype)) - 1), dtype = "ui32", func = func),
      integer(),
      n
    )
    values <- hlo_convert(hlo_and(values, mask), dtype)
  }
  hlo_reshape(values, xla_shape(shape))
}

# ThreeFry-2x32 with 20 rounds (Salmon et al., 2011), as XLA implements it: the
# key `key` and the input `x` are each two `ui32` arrays of the same shape, and
# so is the result.
threefry2x32 <- function(key, x) {
  shp <- shape(x[[1L]]$value_type)
  func <- x[[1L]]$func
  const <- function(value) hlo_broadcast_in_dim(hlo_scalar(value, dtype = "ui32", func = func), integer(), shp)
  rotate_left <- function(v, d) {
    hlo_or(hlo_shift_left(v, const(d)), hlo_shift_right_logical(v, const(32L - d)))
  }
  rotations <- list(c(13L, 15L, 26L, 6L), c(17L, 29L, 16L, 24L))
  rounds <- function(x, r) {
    for (d in rotations[[r]]) {
      x[[1L]] <- hlo_add(x[[1L]], x[[2L]])
      x[[2L]] <- hlo_xor(x[[1L]], rotate_left(x[[2L]], d))
    }
    x
  }

  # 0x1BD11BDA is the parity constant of the key schedule
  ks <- list(key[[1L]], key[[2L]], hlo_xor(hlo_xor(const(0x1BD11BDAL), key[[1L]]), key[[2L]]))
  x[[1L]] <- hlo_add(x[[1L]], ks[[1L]])
  x[[2L]] <- hlo_add(x[[2L]], ks[[2L]])
  for (i in 1:5) {
    x <- rounds(x, (i - 1L) %% 2L + 1L)
    x[[1L]] <- hlo_add(x[[1L]], ks[[i %% 3L + 1L]])
    x[[2L]] <- hlo_add(hlo_add(x[[2L]], ks[[(i + 1L) %% 3L + 1L]]), const(i))
  }
  x
}

# The handler prints its operand as an array of the reversed dimensions, read
# in row-major order. The operand comes in a column-major layout, which is
# exactly that: the row-major reading of the R shape. XLA's compiler crashes on
# a column-major layout for an operand with a size-0 axis, so such an operand,
# which has no elements to order, keeps the default layout.
prim_print[["stablehlo"]] <- function(x, header, footer) {
  xla_shape <- shape(x$value_type)
  layout <- col_major_layout(length(xla_shape))
  if (any(xla_shape == 0L)) {
    layout <- rev(layout)
  }
  backend_config <- stablehlo::CustomOpBackendConfig(list(
    stablehlo::StringAttr(name = "print_header", value = header),
    stablehlo::StringAttr(name = "print_footer", value = footer),
    stablehlo::BoolAttr(name = "reverse_dims", value = TRUE)
  ))

  # has side-effect
  hlo_custom_call(
    x,
    call_target_name = "print_tensor",
    api_version = 4L,
    has_side_effect = TRUE,
    backend_config = backend_config,
    operand_layouts = list(layout),
    result_layouts = list()
  )
  # we just return the input
  list(x)
}

prim_optimization_barrier[["stablehlo"]] <- function(..., output_types) {
  hlo_optimization_barrier(..., simplify = FALSE, output_types = output_types)
}

# higher order primitives --------------------------------------------------------

# The operands after `pred` are the branches' inputs. A StableHLO `if` region
# takes no arguments, so the regions read them by capture.
prim_if[["stablehlo"]] <- function(pred, ..., true, false, n_captures) {
  true_func <- stablehlo(true, id = "", captured = list(...))[[1L]]
  false_func <- stablehlo(false, id = "", captured = list(...))[[1L]]
  hlo_if(pred, true_func, false_func, simplify = FALSE)
}

# The regions take the state as arguments and read the captures past it from
# above.
prim_while[["stablehlo"]] <- function(..., cond, body, n_captures) {
  args <- list(...)
  n_state <- length(args) - n_captures
  state_idx <- seq_len(n_state)
  captured <- args[n_state + seq_len(n_captures)]
  body_func <- stablehlo(body, id = "", captured = captured)[[1L]]
  cond_func <- stablehlo(cond, id = "", captured = captured)[[1L]]
  rlang::exec(hlo_while, !!!args[state_idx], cond = cond_func, body = body_func, simplify = FALSE)
}

# A while loop over the state (i, carry..., out buffers..., xs...). Each
# iteration slices step i of every xs leaf, runs the traced body inline, and
# writes its outputs into the buffers at step i; xs ride along unchanged. The
# step axis is the last R axis, XLA dimension 0, so a step is a contiguous
# block of every buffer and every xs leaf.
prim_scan[["stablehlo"]] <- function(..., body, steps, reverse, n_carry, n_xs, n_captures) {
  args <- list(...)
  outer <- args[[1L]]$func
  n <- as.integer(steps)
  carry0 <- args[seq_len(n_carry)]
  xs0 <- args[n_carry + seq_len(n_xs)]
  avals_body <- lapply(body$outputs, \(out) out$aval)
  out_avals <- avals_body[-seq_len(n_carry)]
  n_out <- length(out_avals)

  # Every step overwrites its own slice, so the buffers only need *some*
  # value; stablehlo has no uninitialized tensor, and zeros are what JAX
  # allocates too. `0L` takes the buffer's data type whatever it is.
  bufs0 <- lapply(out_avals, function(aval) {
    hlo_tensor(
      0L,
      dtype = aval$dtype,
      shape = xla_shape(c(shape(aval), n)),
      func = outer
    )
  })
  # A zero-step scan is the buffers as allocated and the carry untouched. We
  # cannot emit the loop and let the condition stop it immediately: the body
  # slices one step off `xs`, which does not type-check against an `xs` whose
  # last axis is empty, and MLIR verifies a region it never runs.
  if (n == 0L) {
    return(c(carry0, bufs0))
  }

  i0 <- hlo_scalar(0L, dtype = "i32", func = outer)
  state <- c(list(i0), carry0, bufs0, xs0)

  # Both regions declare the full state as their inputs, in state order.
  # `region_input()` names them with auto value ids, which is what keeps one
  # scan lowered inside another's body from redefining the outer region's.
  declare_state <- function() {
    vals <- lapply(state, function(s) {
      vt <- s$value_type
      region_input(as.character(vt$type$dtype), shape(vt))
    })
    list(i = vals[[1L]], rest = vals[-1L])
  }

  cond_func <- stablehlo::local_func("")
  st <- declare_state()
  cond_func <- hlo_return(hlo_compare(
    st$i,
    hlo_scalar(n, dtype = "i32"),
    comparison_direction = "LT",
    compare_type = "SIGNED"
  ))

  body_func <- stablehlo::local_func("")
  st <- declare_state()
  i <- st$i
  carry_in <- st$rest[seq_len(n_carry)]
  bufs_in <- st$rest[n_carry + seq_len(n_out)]
  xs_in <- st$rest[n_carry + n_out + seq_len(n_xs)]
  zero_i <- hlo_scalar(0L, dtype = "i32")
  one_i <- hlo_scalar(1L, dtype = "i32")
  idx <- if (reverse) hlo_subtract(hlo_scalar(n - 1L, dtype = "i32"), i) else i

  # The starts are per XLA dimension, the step's first.
  slices <- lapply(xs_in, function(x) {
    shp <- r_shape(x)
    last <- length(shp)
    starts <- c(list(idx), rep(list(zero_i), last - 1L))
    sl <- rlang::exec(hlo_dynamic_slice, x, !!!starts, slice_sizes = xla_shape(c(shp[-last], 1L)))
    hlo_reshape(sl, xla_shape(shp[-last]))
  })

  env <- HloEnv()
  ins <- c(carry_in, slices, args[n_carry + n_xs + seq_len(n_captures)])
  for (k in seq_along(body$inputs)) {
    env_add(env, body$inputs[[k]], ins[[k]])
  }
  outs <- lower_graph_calls(body, env, stablehlo::.current_func())
  carry_new <- outs[seq_len(n_carry)]
  bufs_new <- Map(
    function(buf, out) {
      shp <- r_shape(out)
      upd <- hlo_reshape(out, xla_shape(c(shp, 1L)))
      starts <- c(list(idx), rep(list(zero_i), length(shp)))
      rlang::exec(hlo_dynamic_update_slice, buf, upd, !!!starts)
    },
    bufs_in,
    outs[n_carry + seq_len(n_out)]
  )
  body_func <- rlang::exec(hlo_return, hlo_add(i, one_i), !!!carry_new, !!!bufs_new, !!!xs_in)

  res <- rlang::exec(hlo_while, !!!state, cond = cond_func, body = body_func, simplify = FALSE)
  c(res[1L + seq_len(n_carry)], res[1L + n_carry + seq_len(n_out)])
}

# The operands are `fn`'s inputs: its arguments, then the captures. Its
# operations are lowered in place. `rule` is not lowered: a gradient is traced,
# as anvl operations, before the program is lowered.
prim_custom_gradient[["stablehlo"]] <- function(..., fn, rule, n_captures) {
  inline_stablehlo_graph(fn, list(...), stablehlo::.current_func())
}

prim_sort[["stablehlo"]] <- function(..., axis, decreasing, stable) {
  ops <- list(...)
  hlo_sort(
    ...,
    dimension = xla_dims(axis, naxes(ops[[1L]])),
    is_stable = stable,
    comparator = .build_sort_comparator(ops, decreasing)
  )
}

# Builds the HLO comparator function consumed by hlo_sort. For float keys we
# use `compare_type = "TOTALORDER"` so NaN bit-patterns sort to the ends
# (vs. IEEE FLOAT, where NaN comparisons return false and positions are
# undefined). We canonicalize -0/+0 → +0 and -NaN/+NaN → +NaN before
# comparing so stable sort treats IEEE-equal values as equal — this keeps
# all NaNs at one end and stops -0/+0 from being silently reordered.
# Mirrors JAX's `_sort_lt_comparator` and `_canonicalize_float_for_sort`.
# `canonicalize = FALSE` drops the -0/-NaN folding, giving the raw TOTALORDER
# that `chlo.top_k` compares with. Only `.hlo_top_k_values()` passes it, so that
# its result does not depend on which lowering the platform took.
.build_sort_comparator <- function(ops, decreasing, canonicalize = TRUE) {
  key_dtype <- ops[[1L]]$value_type$type$dtype
  key_is_float <- is_dtype_float(key_dtype)
  direction <- if (decreasing) "GT" else "LT"

  cmp_func <- stablehlo::local_func("")
  # Declare 2 scalar inputs per sorted array: a_<i>, b_<i>. Only the first pair
  # is compared; the others are declared to match the arity hlo_sort expects.
  pairs <- lapply(seq_along(ops), function(i) {
    dt <- as.character(ops[[i]]$value_type$type$dtype)
    list(a = hlo_input(paste0("a_", i), dt), b = hlo_input(paste0("b_", i), dt))
  })
  a <- pairs[[1L]]$a
  b <- pairs[[1L]]$b

  if (key_is_float && canonicalize) {
    a <- .canonicalize_float_for_sort(a, key_dtype)
    b <- .canonicalize_float_for_sort(b, key_dtype)
  }
  compare_type <- if (key_is_float) {
    "TOTALORDER"
  } else if (is_dtype_int(key_dtype)) {
    "SIGNED"
  } else {
    "UNSIGNED"
  }
  hlo_return(hlo_compare(a, b, comparison_direction = direction, compare_type = compare_type))
}

# Collapse -0 → +0 and -NaN → +NaN on a scalar float. See the comment on
# `.build_sort_comparator` above for why we do this.
.canonicalize_float_for_sort <- function(x, dtype) {
  zero <- hlo_scalar(0L, dtype = dtype, func = x$func)
  canonical_nan <- hlo_scalar(NaN, dtype = dtype, func = x$func)
  is_zero <- hlo_compare(x, zero, comparison_direction = "EQ", compare_type = "FLOAT")
  is_nan <- hlo_compare(x, x, comparison_direction = "NE", compare_type = "FLOAT")
  hlo_select(is_nan, canonical_nan, hlo_select(is_zero, zero, x))
}

# The values-only lowering. Ties may come out in any order, so on CUDA we can
# drop the stability that `chlo.top_k` charges for: it expands there to a
# *stable* sort of (values, iota) plus a slice, and an unstable sort of the
# values alone is never slower than a full sort. XLA's CPU backend lowers the
# CHLO op to a dedicated partial-sort kernel that beats any sort, so it keeps
# it.
#
# The comparator is `chlo.top_k`'s (`canonicalize = FALSE`), not `prim_sort()`'s,
# so the result does not depend on which lowering the platform took -- a
# negative NaN ranks below `-Inf` on both, as `?nv_top_k` documents.
.hlo_top_k_values <- function(x, k) {
  if (!identical(current_platform(), "cuda")) {
    return(hlo_top_k(x, k = k)[[1L]])
  }
  shp <- shape(x$value_type)
  n_axes <- length(shp)
  sorted <- hlo_sort(
    x,
    dimension = n_axes - 1L,
    is_stable = FALSE,
    comparator = .build_sort_comparator(list(x), decreasing = TRUE, canonicalize = FALSE)
  )
  hlo_slice(sorted[[1L]], rep(0L, n_axes), replace(shp, n_axes, k), rep(1L, n_axes))
}

# R axis 1, along which `prim_top_k()` works, is XLA's last dimension, the one
# `hlo_top_k()` works along, so neither lowering moves an axis.
prim_top_k[["stablehlo"]] <- function(x, k, indices, output_types) {
  if (!indices) {
    return(list(.hlo_top_k_values(x, k)))
  }
  out <- hlo_top_k(x, k = k)
  values <- out[[1L]]
  indices <- out[[2L]]

  # `hlo_top_k`'s indices are `i32` by spec, so the shift to 1-based happens
  # there and only the result follows the default integer.
  indices <- hlo_one_based_indices(indices)

  index_dtype <- index_dtype_of(output_types, 2L)
  if (index_dtype != "i32") {
    indices <- hlo_convert(indices, index_dtype)
  }

  list(values, indices)
}

prim_scatter[["stablehlo"]] <- function(
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
  update_op,
  update_fn,
  n_captures,
  ...
) {
  update_func <- stablehlo(update_fn, id = "", captured = list(...))[[1L]]

  n_x <- naxes(x)
  n_indices <- naxes(scatter_indices)
  n_update <- naxes(update)
  batching <- xla_batching_pairs(x_batching_axes, n_x, scatter_indices_batching_axes, n_indices)
  # StableHLO's ScatterDimensionNumbers() follows the spec naming, so the
  # anvl-side argument names are mapped back here. `xla_dims()` turns ascending
  # axes into descending dimensions, and the spec requires `update_window_dims`
  # and `inserted_window_dims` to be sorted (scatter C7, C10).
  scatter_dimension_numbers <- stablehlo::ScatterDimensionNumbers(
    update_window_dims = sort(xla_dims(update_window_axes, n_update)),
    inserted_window_dims = sort(xla_dims(inserted_window_axes, n_x)),
    input_batching_dims = batching[[1L]],
    scatter_indices_batching_dims = batching[[2L]],
    scatter_dims_to_operand_dims = xla_dims(scatter_axes_to_x_axes, n_x),
    index_vector_dim = xla_index_vector_dim(index_vector_axis, n_indices)
  )

  scatter_indices_0based <- hlo_zero_based_indices(scatter_indices)

  result <- hlo_scatter(
    inputs = list(x),
    scatter_indices = scatter_indices_0based,
    updates = list(update),
    scatter_dimension_numbers = scatter_dimension_numbers,
    indices_are_sorted = indices_are_sorted,
    unique_indices = unique_indices,
    update_computation = update_func
  )

  list(result)
}

prim_gather[["stablehlo"]] <- function(
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
  n_x <- naxes(x)
  n_indices <- naxes(start_indices)
  # the result has the offset axes and one batch axis per axis of the indices
  # other than the index vector axis
  n_out <- length(offset_axes) + n_indices - (index_vector_axis <= n_indices)
  batching <- xla_batching_pairs(x_batching_axes, n_x, start_indices_batching_axes, n_indices)
  # StableHLO's GatherDimensionNumbers() follows the spec naming, so the
  # anvl-side `x_batching_axes` is mapped back here. `xla_dims()` turns
  # ascending axes into descending dimensions, and the spec requires
  # `offset_dims` and `collapsed_slice_dims` to be sorted (gather C4, C7).
  gdn <- stablehlo::GatherDimensionNumbers(
    offset_dims = sort(xla_dims(offset_axes, n_out)),
    collapsed_slice_dims = sort(xla_dims(collapsed_slice_axes, n_x)),
    operand_batching_dims = batching[[1L]],
    start_indices_batching_dims = batching[[2L]],
    start_index_map = xla_dims(start_index_map, n_x),
    index_vector_dim = xla_index_vector_dim(index_vector_axis, n_indices)
  )

  start_indices_0based <- hlo_zero_based_indices(start_indices)

  result <- hlo_gather(
    x,
    start_indices_0based,
    gather_dimension_numbers = gdn,
    slice_sizes = xla_per_axis(slice_sizes),
    indices_are_sorted = indices_are_sorted
  )

  list(result)
}

# pjrt's `potrf` custom call (LAPACK on the host, cuSOLVER on CUDA) factors
# column-major matrices, batched over its leading dimensions, and returns
# potrf's `info` per matrix, which is non-zero for a matrix that is not
# positive definite. The trailing batch axes are XLA's leading dimensions, and
# the matrix in R axes 1 and 2 is the block behind them, (column, row) in XLA
# order: the column-major matrix as XLA's row-major bytes. So potrf factors
# exactly the matrix the R axes hold and writes the factor the same way.
prim_chol[["stablehlo"]] <- function(x, lower) {
  op_shape <- shape(x$value_type)
  r <- length(op_shape)
  batch_dims <- seq_len(r - 2L) - 1L
  dt <- dtype(x)

  row_major <- rev(seq_len(r) - 1L)
  out <- hlo_custom_call(
    x,
    call_target_name = "potrf",
    api_version = 4L,
    has_side_effect = FALSE,
    backend_config = stablehlo::CustomOpBackendConfig(list(
      stablehlo::BoolAttr(name = "lower", value = lower)
    )),
    output_types = list(vt(dtype = dt, shape = op_shape), vt(dtype = "i32", shape = op_shape[seq_len(r - 2L)])),
    operand_layouts = list(row_major),
    result_layouts = list(row_major, rev(batch_dims))
  )
  L <- out[[1L]]
  info <- out[[2L]]

  # A matrix that is not positive definite factors to NaN, as XLA's own
  # expansion of `stablehlo.cholesky` does.
  ok <- hlo_compare(
    info,
    hlo_tensor(0L, dtype = "i32", shape = shape(info$value_type), func = x$func),
    comparison_direction = "EQ",
    compare_type = "SIGNED"
  )
  ok <- hlo_broadcast_in_dim(ok, batch_dims, op_shape)
  L <- hlo_select(ok, L, hlo_tensor(NaN, dtype = dt, shape = op_shape, func = x$func))

  # potrf leaves the input's values in the other triangle: zero it, so
  # downstream code (including pullback rules) never sees them. A block's first
  # dimension is the matrix's column, its second the row.
  n <- op_shape[r]
  mat_shape <- c(n, n)
  cols <- hlo_iota(iota_dimension = 0L, dtype = "i32", shape = mat_shape, func = x$func)
  rows <- hlo_iota(iota_dimension = 1L, dtype = "i32", shape = mat_shape, func = x$func)
  mask <- hlo_compare(rows, cols, comparison_direction = if (lower) "GE" else "LE", compare_type = "SIGNED")
  if (r > 2L) {
    mask <- hlo_broadcast_in_dim(mask, c(r - 2L, r - 1L), op_shape)
  }
  zero <- hlo_tensor(0L, dtype = dt, shape = op_shape, func = x$func)
  list(hlo_select(mask, L, zero))
}

# XLA lowers `stablehlo.triangular_solve` to cuBLAS trsm on CUDA, but on the
# CPU expands it into blocked HLO that is several times slower than BLAS trsm.
# So, as JAX does, the CPU takes pjrt's `trsm` custom call for the data types
# BLAS has.
#
# Both take their batch dimensions first and solve on the blocks behind them.
# The trailing batch axes are XLA's leading dimensions, and each block holds the
# matrix in R axes 1 and 2 transposed, as (column, row). op(a) x = b is
# t(x) op(t(a)) = t(b), so they solve for t(x) with the side and the triangle
# swapped.
prim_triangular_solve[["stablehlo"]] <- function(a, b, left_side, lower, unit_diagonal, transpose_a, output_types) {
  if (identical(current_platform(), "cpu") && as.character(dtype(a)) %in% c("f32", "f64")) {
    row_major <- rev(seq_len(naxes(b)) - 1L)
    x <- hlo_custom_call(
      a,
      b,
      call_target_name = "trsm",
      api_version = 4L,
      has_side_effect = FALSE,
      backend_config = stablehlo::CustomOpBackendConfig(list(
        stablehlo::BoolAttr(name = "left_side", value = !left_side),
        stablehlo::BoolAttr(name = "lower", value = !lower),
        stablehlo::BoolAttr(name = "transpose_a", value = transpose_a),
        stablehlo::BoolAttr(name = "unit_diagonal", value = unit_diagonal)
      )),
      output_types = output_types,
      operand_layouts = list(row_major, row_major),
      result_layouts = list(row_major)
    )
  } else {
    x <- hlo_triangular_solve(
      a,
      b,
      left_side = !left_side,
      lower = !lower,
      unit_diagonal = unit_diagonal,
      transpose_a = if (transpose_a) "TRANSPOSE" else "NO_TRANSPOSE",
      output_types = output_types
    )
  }
  list(x)
}

# The 2-D decompositions run pjrt handlers that take an m x n matrix as its
# column-major data, a row-major [n, m] buffer: the XLA shape and layout the
# matrix has anyway, so the operand and the results pass without a transpose.
# The shapes below are XLA shapes, the R shapes reversed.
prim_qr[["stablehlo"]] <- function(x) {
  dt <- x$value_type$type$dtype
  n <- shape(x)[1L]
  m <- shape(x)[2L]
  k <- min(m, n)

  # pjrt's QR is split across two custom calls (matching jaxlib + LAPACK):
  #   geqrf(A) -> packed (m, n), tau (k)
  #     where R is encoded in the upper triangle of the first k rows of packed
  #     and the Householder reflectors live in the strict lower triangle.
  #   orgqr(packed, tau) -> Q (m, k)
  # We compose them here and reconstruct R from packed via slice + triu.
  geqrf_out <- hlo_custom_call(
    x,
    call_target_name = "geqrf",
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = list(vt(dtype = dt, shape = c(n, m)), vt(dtype = dt, shape = k)),
    operand_layouts = row_major_layouts(2L),
    result_layouts = row_major_layouts(2L, 1L)
  )
  packed <- geqrf_out[[1L]]
  tau <- geqrf_out[[2L]]

  Q <- hlo_custom_call(
    packed,
    tau,
    call_target_name = "orgqr",
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = list(vt(dtype = dt, shape = c(k, m))),
    operand_layouts = row_major_layouts(2L, 1L),
    result_layouts = row_major_layouts(2L)
  )

  # R is the upper triangle of the first k rows of packed, which run along XLA
  # dimension 1. Slice if m > k (= m > n); when m <= n, k = m and packed
  # already has the right row count.
  packed_top <- if (m > k) {
    hlo_slice(
      packed,
      start_indices = c(0L, 0L),
      limit_indices = c(n, k),
      strides = c(1L, 1L)
    )
  } else {
    packed
  }
  top_shape <- c(n, k)
  cols <- hlo_iota(iota_dimension = 0L, dtype = "i32", shape = top_shape, func = x$func)
  rows <- hlo_iota(iota_dimension = 1L, dtype = "i32", shape = top_shape, func = x$func)
  triu_mask <- hlo_compare(rows, cols, comparison_direction = "LE", compare_type = "SIGNED")
  zero <- hlo_tensor(0L, dtype = dt, shape = top_shape)
  R <- hlo_select(triu_mask, packed_top, zero)

  list(Q, R)
}

# Convert a LAPACK pivot vector (length `k`, 1-based sequential row swaps
# from getrf) into a length-`n` 1-based permutation vector, using
# `hlo_while` + dynamic slice / update_slice. The body Func closes over the
# outer `pivots` Value the same way an anvl-traced `prim_while` body would.
pivots_to_permutation <- function(pivots, n) {
  func <- pivots$func
  k <- shape(pivots$value_type)[[1L]]

  init_perm <- hlo_one_based_indices(hlo_iota(iota_dimension = 0L, dtype = "i32", shape = n, func = func))
  init_i <- hlo_scalar(0L, dtype = "i32", func = func)

  cond_func <- stablehlo::local_func("")
  i <- region_input("i32")
  region_input("i32", n) # unused in cond; declared to match state shape
  cond_func <- hlo_return(hlo_compare(
    i,
    hlo_scalar(k, dtype = "i32"),
    comparison_direction = "LT",
    compare_type = "SIGNED"
  ))

  # Body of one loop iteration (i in 0..k-1): LAPACK getrf swapped row i
  # with row pivots[i]; we mirror that on `perm` by exchanging perm[i] and
  # perm[j], where j = pivots[i] - 1 converts pivots' 1-based value to a
  # 0-based index.
  body_func <- stablehlo::local_func("")
  i <- region_input("i32")
  perm <- region_input("i32", n)
  # constant in the region
  pivots_in_body <- stablehlo::FuncValue(
    pivots$value_id,
    pivots$value_type,
    stablehlo::.current_func()
  )
  one <- hlo_scalar(1L, dtype = "i32")
  pivots_i <- hlo_reshape(hlo_dynamic_slice(pivots_in_body, i, slice_sizes = 1L), integer())
  # pjrt custom call returns pivots 1-based (cuSolve matches LAPACK), so we have to convert
  # because stablehlo is 0-based
  j <- hlo_subtract(pivots_i, one) # 0-based swap target
  val_i <- hlo_dynamic_slice(perm, i, slice_sizes = 1L) # perm[i]
  val_j <- hlo_dynamic_slice(perm, j, slice_sizes = 1L) # perm[j]
  new_perm <- hlo_dynamic_update_slice(perm, val_j, i) # perm[i] <- val_j
  new_perm <- hlo_dynamic_update_slice(new_perm, val_i, j) # perm[j] <- val_i
  body_func <- hlo_return(hlo_add(i, one), new_perm)

  hlo_while(
    init_i,
    init_perm,
    cond = cond_func,
    body = body_func,
    simplify = FALSE
  )[[2L]]
}

prim_lu[["stablehlo"]] <- function(x, output_types) {
  dt <- x$value_type$type$dtype
  n <- shape(x)[1L]
  m <- shape(x)[2L]
  k <- min(m, n)

  # LAPACK's `getrf` (and cuSOLVER, which matches it) writes `ipiv` as 32-bit
  # integers, so the custom call and the loop that walks it stay `i32`
  # whatever the default integer is; only the two outputs are converted.
  index_dtype <- index_dtype_of(output_types, 2L)

  out <- hlo_custom_call(
    x,
    call_target_name = "lu",
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = list(vt(dtype = dt, shape = c(n, m)), vt(dtype = "i32", shape = k)),
    operand_layouts = row_major_layouts(2L),
    result_layouts = row_major_layouts(2L, 1L)
  )
  LU <- out[[1L]]
  pivots <- out[[2L]]
  permutation <- pivots_to_permutation(pivots, m)
  if (index_dtype != "i32") {
    pivots <- hlo_convert(pivots, index_dtype)
    permutation <- hlo_convert(permutation, index_dtype)
  }
  list(LU, pivots, permutation)
}

prim_svd[["stablehlo"]] <- function(x) {
  dt <- x$value_type$type$dtype
  n <- shape(x)[1L]
  m <- shape(x)[2L]
  k <- min(m, n)

  # U is (m, k) and Vt is (k, n).
  u_type <- vt(dtype = dt, shape = c(k, m))
  s_type <- vt(dtype = dt, shape = k)
  vt_type <- vt(dtype = dt, shape = c(n, k))

  # cuSOLVER's gesvd requires m >= n. When m < n on CUDA we mirror JAX's
  # layout-flip trick: declare x / U / Vt column-major instead of row-major
  # and tell the FFI handler `transposed = TRUE`. A column-major [n, m] buffer
  # holds the column-major data of A^T, so cuSOLVER (with m and n swapped)
  # computes A^T = V S U^T and writes its U / Vt outputs straight into the
  # Vt / U slots, whose column-major layouts transpose them back. Unlike the
  # row-major ones, these layouts are a physical transpose of x and of both
  # outputs. The CUDA handler always reads a `transposed` attribute, so we
  # emit it (with both values) whenever targeting CUDA; other platforms (host
  # LAPACK gesdd handles any shape) take the plain path with no
  # backend_config.
  on_cuda <- identical(current_platform(), "cuda")
  transposed <- on_cuda && m < n

  backend_config <- if (on_cuda) {
    stablehlo::CustomOpBackendConfig(list(
      stablehlo::BoolAttr(name = "transposed", value = transposed)
    ))
  }
  layouts <- if (transposed) col_major_layouts else row_major_layouts
  out <- hlo_custom_call(
    x,
    call_target_name = "svd",
    api_version = 4L,
    has_side_effect = FALSE,
    backend_config = backend_config,
    output_types = list(u_type, s_type, vt_type),
    operand_layouts = layouts(2L),
    result_layouts = layouts(2L, 1L, 2L)
  )
  list(out[[2L]], out[[1L]], out[[3L]])
}

prim_eigh[["stablehlo"]] <- function(x) {
  dt <- x$value_type$type$dtype
  n <- shape(x)[1L]

  # The "eigh" custom call returns (vectors, values) positionally. We
  # re-bundle to (values, vectors) matching `base::eigen()`.
  out <- hlo_custom_call(
    x,
    call_target_name = "eigh",
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = list(vt(dtype = dt, shape = c(n, n)), vt(dtype = dt, shape = n)),
    operand_layouts = row_major_layouts(2L),
    result_layouts = row_major_layouts(2L, 1L)
  )
  list(out[[2L]], out[[1L]]) # values, vectors
}

prim_convolution[["stablehlo"]] <- function(
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
  # All three arrays have the same number of axes. The spatial lists keep their
  # order: the strides, padding and dilations pair with them by position.
  n <- naxes(x)
  shlo_dn <- stablehlo::ConvDimensionNumbers(
    input_batch_dimension = xla_dims(x_batch_axis, n),
    input_feature_dimension = xla_dims(x_feature_axis, n),
    input_spatial_dimensions = xla_dims(x_spatial_axes, n),
    kernel_input_feature_dimension = xla_dims(kernel_input_feature_axis, n),
    kernel_output_feature_dimension = xla_dims(kernel_output_feature_axis, n),
    kernel_spatial_dimensions = xla_dims(kernel_spatial_axes, n),
    output_batch_dimension = xla_dims(output_batch_axis, n),
    output_feature_dimension = xla_dims(output_feature_axis, n),
    output_spatial_dimensions = xla_dims(output_spatial_axes, n)
  )
  list(hlo_convolution(
    x,
    kernel,
    dimension_numbers = shlo_dn,
    window_strides = window_strides,
    padding = padding,
    lhs_dilation = x_dilation,
    rhs_dilation = kernel_dilation,
    feature_group_count = feature_group_count,
    batch_group_count = batch_group_count,
    precision_config = rep(toupper(precision), 2L)
  ))
}
