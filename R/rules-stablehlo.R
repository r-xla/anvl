#' @include primitives.R

# Type inference was already done at trace time. Rules that declare an
# `output_types` parameter receive the known result types from the lowering
# driver (see stablehlo()) and forward them to their hlo_* builder, letting
# stablehlo skip re-inference. Composite and region rules (reduce, cumulative
# ops, linalg decompositions, ...) omit the parameter and keep normal inference.

prim_fill[["stablehlo"]] <- function(value, shape, dtype) {
  list(hlo_tensor(value, shape = shape, dtype = dtype))
}

prim_add[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_add(lhs, rhs, output_types = output_types))
}

prim_mul[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_multiply(lhs, rhs, output_types = output_types))
}

prim_sub[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_subtract(lhs, rhs, output_types = output_types))
}

prim_negate[["stablehlo"]] <- function(x, output_types) {
  list(hlo_negate(x, output_types = output_types))
}

prim_div[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_divide(lhs, rhs, output_types = output_types))
}

prim_pow[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_power(lhs, rhs, output_types = output_types))
}

prim_broadcast_in_axes[["stablehlo"]] <- function(x, shape, broadcast_axes, output_types) {
  list(hlo_broadcast_in_dim(x, broadcast_axes - 1L, shape, output_types = output_types))
}

prim_dot_general[["stablehlo"]] <- function(lhs, rhs, contracting_axes, batching_axes, precision, output_types) {
  contracting_axes <- lapply(contracting_axes, \(x) x - 1L)
  batching_axes <- lapply(batching_axes, \(x) x - 1L)
  list(hlo_dot_general(
    lhs,
    rhs,
    contracting_axes,
    batching_axes,
    precision_config = toupper(precision),
    output_types = output_types
  ))
}

prim_transpose[["stablehlo"]] <- function(x, permutation, output_types) {
  list(hlo_transpose(x, permutation - 1L, output_types = output_types))
}

prim_reshape[["stablehlo"]] <- function(x, shape, output_types) {
  list(hlo_reshape(x, shape, output_types = output_types))
}

prim_concatenate[["stablehlo"]] <- function(..., axis, output_types) {
  list(hlo_concatenate(..., dimension = axis - 1L, output_types = output_types))
}

prim_static_slice[["stablehlo"]] <- function(x, start_indices, limit_indices, strides, output_types) {
  # we use 1:n, which includes n, but this translates to 0:n in stablehlo
  list(hlo_slice(x, start_indices - 1L, limit_indices, strides, output_types = output_types))
}

prim_dynamic_slice[["stablehlo"]] <- function(x, ..., slice_sizes, output_types) {
  start_indices <- list(...)
  # Convert start indices from 1-based to 0-based by subtracting 1
  start_indices_0based <- lapply(start_indices, function(idx) {
    one <- hlo_scalar(1L, dtype = dtype(idx), func = idx$func)
    hlo_subtract(idx, one)
  })
  list(rlang::exec(
    hlo_dynamic_slice,
    x,
    !!!start_indices_0based,
    slice_sizes = slice_sizes,
    output_types = output_types
  ))
}

prim_dynamic_update_slice[["stablehlo"]] <- function(x, update, ..., output_types) {
  start_indices <- list(...)
  if (!length(start_indices)) {
    return(list(update))
  }
  # Convert start indices from 1-based to 0-based by subtracting 1
  start_indices_0based <- lapply(start_indices, function(idx) {
    one <- hlo_scalar(1L, dtype = dtype(idx), func = idx$func)
    hlo_subtract(idx, one)
  })
  list(rlang::exec(
    hlo_dynamic_update_slice,
    x,
    update,
    !!!start_indices_0based,
    output_types = output_types
  ))
}


.stablehlo_apply_reduce <- function(reductor, x, init, axes, drop) {
  local_func("")
  dt <- as.character(x$value_type$type$dtype)
  f <- hlo_return(reductor(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))
  out <- hlo_reduce(list(x), list(init(x)), axes - 1L, f)

  if (drop) {
    return(list(out))
  }

  shape_out <- shape(x$value_type)
  shape_out[axes] <- 1L
  list(hlo_reshape(out, shape_out))
}

prim_reduce_sum[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    hlo_scalar(0, dtype = dtype(x), func = x$func)
  }
  .stablehlo_apply_reduce(hlo_add, x, init, axes, drop)
}

prim_reduce_prod[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    hlo_scalar(1, dtype = dtype(x), func = x$func)
  }
  .stablehlo_apply_reduce(hlo_multiply, x, init, axes, drop)
}


prim_reduce_max[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_minval(dtype(x), "cpu"))
  }
  .stablehlo_apply_reduce(hlo_maximum, x, init, axes, drop)
}

prim_reduce_min[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    # platform does not matter when we just embed the init value in stablehlo
    hlo_scalar(nv_maxval(dtype(x), "cpu"))
  }
  .stablehlo_apply_reduce(hlo_minimum, x, init, axes, drop)
}

prim_reduce_any[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    hlo_scalar(FALSE)
  }
  .stablehlo_apply_reduce(hlo_or, x, init, axes, drop)
}

prim_reduce_all[["stablehlo"]] <- function(x, axes, drop) {
  init <- function(x) {
    hlo_scalar(TRUE)
  }
  .stablehlo_apply_reduce(hlo_and, x, init, axes, drop)
}

# XLA compiler optimizes this according to JAX comment
# (there apparently were differences between {C,G,T}PU backend, not no longer it seems)
.stablehlo_apply_cum <- function(reductor, x, init, axis) {
  shp <- shape(x)
  rank <- length(shp)
  s_d <- shp[[axis]]
  window_dimensions <- rep(1L, rank)
  window_dimensions[[axis]] <- s_d
  ones <- rep(1L, rank)
  padding <- matrix(0L, nrow = rank, ncol = 2L)
  padding[axis, 1L] <- s_d - 1L

  local_func("")
  dt <- as.character(x$value_type$type$dtype)
  body <- hlo_return(reductor(
    hlo_input("x", dt),
    hlo_input("y", dt)
  ))

  list(hlo_reduce_window(
    inputs = x,
    init_values = init(x),
    window_dimensions = window_dimensions,
    window_strides = ones,
    base_dilations = ones,
    window_dilations = ones,
    padding = padding,
    body = body
  ))
}

prim_cumsum[["stablehlo"]] <- function(x, axis) {
  init <- function(x) {
    hlo_scalar(0, dtype = dtype(x), func = x$func)
  }
  .stablehlo_apply_cum(hlo_add, x, init, axis)
}

prim_cumprod[["stablehlo"]] <- function(x, axis) {
  init <- function(x) {
    hlo_scalar(1, dtype = dtype(x), func = x$func)
  }
  .stablehlo_apply_cum(hlo_multiply, x, init, axis)
}

# Here we also return the indices to simplify reverse rule
.stablehlo_apply_cum_extreme <- function(x, axis, is_max) {
  shp <- shape(x)
  rank <- length(shp)
  s_d <- shp[[axis]]
  window_dimensions <- rep(1L, rank)
  window_dimensions[[axis]] <- s_d
  ones <- rep(1L, rank)
  padding <- matrix(0L, nrow = rank, ncol = 2L)
  padding[axis, 1L] <- s_d - 1L

  v_dtype <- as.character(x$value_type$type$dtype)
  iota <- hlo_iota(iota_dimension = axis - 1L, dtype = "i32", shape = shp)

  init_v_fn <- if (is_max) nv_minval else nv_maxval
  init_v <- hlo_scalar(init_v_fn(v_dtype, "cpu"))
  init_i <- hlo_scalar(0L, dtype = "i32", func = x$func)

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
  reductor <- if (is_float) {
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
  body <- .r_reductor_to_hlo_func(
    reductor,
    list(
      lv = nv_aval(v_dtype, integer()),
      li = nv_aval("i32", integer()),
      rv = nv_aval(v_dtype, integer()),
      ri = nv_aval("i32", integer())
    )
  )

  out <- hlo_reduce_window(
    inputs = list(x, iota),
    init_values = list(init_v, init_i),
    window_dimensions = window_dimensions,
    window_strides = ones,
    base_dilations = ones,
    window_dilations = ones,
    padding = padding,
    body = body
  )
  values <- out[[1L]]
  indices_0 <- out[[2L]]
  one <- hlo_scalar(1L, dtype = "i32", func = indices_0$func)
  one_bc <- hlo_broadcast_in_dim(one, integer(0), shape(indices_0$value_type))
  list(values, hlo_add(indices_0, one_bc))
}

prim_cummax[["stablehlo"]] <- function(x, axis) {
  .stablehlo_apply_cum_extreme(x, axis, is_max = TRUE)
}

prim_cummin[["stablehlo"]] <- function(x, axis) {
  .stablehlo_apply_cum_extreme(x, axis, is_max = FALSE)
}

prim_reduce[["stablehlo"]] <- function(x, init, axes, drop, reductor_graph, .env) {
  red_func <- stablehlo(reductor_graph, id = "", constants_as_inputs = FALSE, env = .env)[[1L]]
  out <- hlo_reduce(
    inputs = list(x),
    init_values = list(init),
    dimensions = axes - 1L,
    body = red_func
  )
  if (drop) {
    return(list(out))
  }
  shape_out <- shape(x$value_type)
  shape_out[axes] <- 1L
  list(hlo_reshape(out, shape_out))
}

.r_reductor_to_hlo_func <- function(fn, dummy_args) {
  graph <- trace_fn(fn, dummy_args, desc = local_descriptor())
  stablehlo(graph, id = "", constants_as_inputs = FALSE)[[1L]]
}

.stablehlo_arg_extreme <- function(x, axis, drop, direction, init_v_fn) {
  shp <- shape(x$value_type)
  v_dtype <- x$value_type$type$dtype
  iota <- hlo_iota(iota_dimension = axis - 1L, dtype = "i32", shape = shp)
  init_v <- hlo_scalar(init_v_fn(v_dtype, "cpu"))
  init_i <- hlo_scalar(0L, dtype = "i32", func = x$func)

  # Reductor: pick lhs unless rhs is strictly better. On a tie, pick the
  # smaller index. XLA's reduce is associative but may be applied in any order
  # (and is not commutative), so we cannot assume li < ri -- on CUDA the
  # accumulator index can exceed the incoming index. We therefore break ties
  # explicitly on `ri < li` rather than relying on argument order.
  cmp <- if (direction == "GT") prim_gt else prim_lt
  reductor <- function(lv, li, rv, ri) {
    rhs_better <- nv_or(cmp(rv, lv), nv_and(prim_eq(rv, lv), prim_lt(ri, li)))
    list(nv_ifelse(rhs_better, rv, lv), nv_ifelse(rhs_better, ri, li))
  }
  body <- .r_reductor_to_hlo_func(
    reductor,
    list(
      lv = nv_aval(v_dtype, integer()),
      li = nv_aval("i32", integer()),
      rv = nv_aval(v_dtype, integer()),
      ri = nv_aval("i32", integer())
    )
  )

  out <- hlo_reduce(
    inputs = list(x, iota),
    init_values = list(init_v, init_i),
    dimensions = axis - 1L,
    body = body
  )
  # convert to 1-based
  result <- out[[2L]]
  one <- hlo_scalar(1L, dtype = "i32", func = result$func)
  one_bc <- hlo_broadcast_in_dim(one, integer(0), shape(result$value_type))
  result <- hlo_add(result, one_bc)
  if (drop) {
    return(list(result))
  }
  shape_out <- shp
  shape_out[axis] <- 1L
  list(hlo_reshape(result, shape_out))
}

prim_argmax[["stablehlo"]] <- function(x, axis, drop) {
  .stablehlo_arg_extreme(x, axis, drop, direction = "GT", init_v_fn = nv_minval)
}

prim_argmin[["stablehlo"]] <- function(x, axis, drop) {
  .stablehlo_arg_extreme(x, axis, drop, direction = "LT", init_v_fn = nv_maxval)
}

# comparison jit rules ----------------------------------------------------------

.compare_type_for <- function(vt) {
  dt <- vt$value_type$type$dtype
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
  function(lhs, rhs, output_types) {
    ct <- .compare_type_for(lhs)
    list(hlo_compare(
      lhs,
      rhs,
      comparison_direction = direction,
      compare_type = ct,
      output_types = output_types
    ))
  }
}

prim_eq[["stablehlo"]] <- .stablehlo_compare_bin("EQ")
prim_ne[["stablehlo"]] <- .stablehlo_compare_bin("NE")
prim_gt[["stablehlo"]] <- .stablehlo_compare_bin("GT")
prim_ge[["stablehlo"]] <- .stablehlo_compare_bin("GE")
prim_lt[["stablehlo"]] <- .stablehlo_compare_bin("LT")
prim_le[["stablehlo"]] <- .stablehlo_compare_bin("LE")


# binary simple math jit rules ---------------------------------------------------

prim_max[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_maximum(lhs, rhs, output_types = output_types))
}

prim_min[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_minimum(lhs, rhs, output_types = output_types))
}

prim_remainder[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_remainder(lhs, rhs, output_types = output_types))
}

prim_and[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_and(lhs, rhs, output_types = output_types))
}

prim_not[["stablehlo"]] <- function(x, output_types) {
  list(hlo_not(x, output_types = output_types))
}

prim_or[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_or(lhs, rhs, output_types = output_types))
}

prim_xor[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_xor(lhs, rhs, output_types = output_types))
}

prim_shift_left[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_shift_left(lhs, rhs, output_types = output_types))
}

prim_shift_right_logical[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_shift_right_logical(lhs, rhs, output_types = output_types))
}

prim_shift_right_arithmetic[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_shift_right_arithmetic(lhs, rhs, output_types = output_types))
}

prim_atan2[["stablehlo"]] <- function(lhs, rhs, output_types) {
  list(hlo_atan2(lhs, rhs, output_types = output_types))
}

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

prim_ceil[["stablehlo"]] <- function(x, output_types) {
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

prim_logistic[["stablehlo"]] <- function(x, output_types) {
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

prim_polygamma[["stablehlo"]] <- function(n, x, output_types) {
  list(hlo_polygamma(n, x, output_types = output_types))
}

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

prim_clamp[["stablehlo"]] <- function(min_val, x, max_val, output_types) {
  list(hlo_clamp(min_val, x, max_val, output_types = output_types))
}

prim_reverse[["stablehlo"]] <- function(x, axes, output_types) {
  list(hlo_reverse(x, axes - 1L, output_types = output_types))
}

prim_iota[["stablehlo"]] <- function(axis, dtype, shape, start) {
  out <- hlo_iota(iota_dimension = axis - 1L, dtype = dtype, shape = shape)
  if (start != 0L) {
    offset <- hlo_broadcast_in_dim(
      hlo_scalar(start, dtype = dtype, func = out$func),
      integer(0),
      shape
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
    edge_padding_low,
    edge_padding_high,
    interior_padding,
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


prim_ifelse[["stablehlo"]] <- function(pred, true_value, false_value, output_types) {
  list(hlo_select(pred, true_value, false_value, output_types = output_types))
}

# RNG jit rules --------------------------------------------------------

prim_rng_bit_generator[["stablehlo"]] <- function(initial_state, rng_algorithm, dtype, shape) {
  hlo_rng_bit_generator(initial_state, rng_algorithm, dtype, shape)
}

prim_print[["stablehlo"]] <- function(x, footer) {
  backend_config <- stablehlo::CustomOpBackendConfig(list(
    stablehlo::StringAttr(name = "print_header", value = "AnvlArray"),
    stablehlo::StringAttr(name = "print_footer", value = footer)
  ))

  # row-major layout in minor-to-major order: [N-1, ..., 1, 0]
  row_major <- rev(seq_len(length(shape(x))) - 1L)

  # has side-effect
  hlo_custom_call(
    x,
    call_target_name = "print_tensor",
    api_version = 4L,
    has_side_effect = TRUE,
    backend_config = backend_config,
    operand_layouts = list(row_major),
    result_layouts = list()
  )
  # we just return the input
  list(x)
}

# higher order primitives --------------------------------------------------------

prim_if[["stablehlo"]] <- function(pred, true_graph, false_graph, .env) {
  true_func <- stablehlo(true_graph, id = "", constants_as_inputs = FALSE, env = .env)[[1L]]
  false_func <- stablehlo(false_graph, id = "", constants_as_inputs = FALSE, env = .env)[[1L]]
  hlo_if(pred, true_func, false_func, simplify = FALSE)
}

prim_while[["stablehlo"]] <- function(..., cond_graph, body_graph, .env) {
  body_func <- stablehlo(body_graph, id = "", constants_as_inputs = FALSE, env = .env)[[1L]]
  cond_func <- stablehlo(cond_graph, id = "", constants_as_inputs = FALSE, env = .env)[[1L]]
  hlo_while(..., cond = cond_func, body = body_func, simplify = FALSE)
}

prim_sort[["stablehlo"]] <- function(..., axis, descending, is_stable) {
  ops <- list(...)
  hlo_sort(
    ...,
    dimension = axis - 1L,
    is_stable = is_stable,
    comparator = .build_sort_comparator(ops, descending)
  )
}

# Builds the HLO comparator function consumed by hlo_sort. For float keys we
# use `compare_type = "TOTALORDER"` so NaN bit-patterns sort to the ends
# (vs. IEEE FLOAT, where NaN comparisons return false and positions are
# undefined). We canonicalize -0/+0 → +0 and -NaN/+NaN → +NaN before
# comparing so stable sort treats IEEE-equal values as equal — this keeps
# all NaNs at one end and stops -0/+0 from being silently reordered.
# Mirrors JAX _sort_lt_comparator, _canonicalize_float_for_sort).
.build_sort_comparator <- function(ops, descending) {
  key_dtype <- ops[[1L]]$value_type$type$dtype
  key_is_float <- is_dtype_float(key_dtype)
  direction <- if (descending) "GT" else "LT"

  cmp_func <- stablehlo::local_func("")
  # Declare 2 scalar inputs per sorted array: a_<i>, b_<i>. We keep references to
  # the first pair only; subsequent inputs are declared (to match the arity
  # hlo_sort expects) but ignored.
  a <- NULL
  b <- NULL
  for (i in seq_along(ops)) {
    dt <- as.character(ops[[i]]$value_type$type$dtype)
    ai <- hlo_input(paste0("a_", i), dt)
    bi <- hlo_input(paste0("b_", i), dt)
    if (i == 1L) {
      a <- ai
      b <- bi
    }
  }

  if (key_is_float) {
    a <- .canonicalize_float_for_sort(a, key_dtype)
    b <- .canonicalize_float_for_sort(b, key_dtype)
    result <- hlo_compare(a, b, comparison_direction = direction, compare_type = "TOTALORDER")
  } else {
    ct <- if (is_dtype_int(key_dtype)) "SIGNED" else "UNSIGNED"
    result <- hlo_compare(a, b, comparison_direction = direction, compare_type = ct)
  }
  hlo_return(result)
}

# Collapse -0 → +0 and -NaN → +NaN on a scalar float. See the comment on
# `.build_sort_comparator` above for why we do this.
.canonicalize_float_for_sort <- function(x, dtype) {
  zero <- hlo_scalar(0, dtype = dtype, func = x$func)
  canonical_nan <- hlo_scalar(NaN, dtype = dtype, func = x$func)
  is_zero <- hlo_compare(x, zero, comparison_direction = "EQ", compare_type = "FLOAT")
  is_nan <- hlo_compare(x, x, comparison_direction = "NE", compare_type = "FLOAT")
  hlo_select(is_nan, canonical_nan, hlo_select(is_zero, zero, x))
}

prim_top_k[["stablehlo"]] <- function(x, k) {
  out <- hlo_top_k(x, k = k)
  values <- out[[1L]]
  indices <- out[[2L]]

  # to 1-based
  one <- hlo_scalar(1L, dtype = "i32", func = indices$func)
  one_bc <- hlo_broadcast_in_dim(one, integer(0), shape(indices$value_type))
  indices <- hlo_add(indices, one_bc)

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
  update_computation_graph,
  .env
) {
  update_func <- stablehlo(update_computation_graph, id = "", constants_as_inputs = FALSE, env = .env)[[1L]]

  # StableHLO's ScatterDimensionNumbers() follows the spec naming, so the
  # anvl-side argument names are mapped back here.
  scatter_dimension_numbers <- stablehlo::ScatterDimensionNumbers(
    update_window_dims = update_window_axes - 1L,
    inserted_window_dims = inserted_window_axes - 1L,
    input_batching_dims = x_batching_axes - 1L,
    scatter_indices_batching_dims = scatter_indices_batching_axes - 1L,
    scatter_dims_to_operand_dims = scatter_axes_to_x_axes - 1L,
    index_vector_dim = index_vector_axis - 1L
  )

  one <- hlo_tensor(1L, shape = shape(scatter_indices), dtype = dtype(scatter_indices))
  scatter_indices_0based <- hlo_subtract(scatter_indices, one)

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
  # Convert 1-based axis numbers to 0-based for stablehlo
  # StableHLO's GatherDimensionNumbers() follows the spec naming, so the
  # anvl-side `x_batching_axes` is mapped back here.
  gdn_0based <- stablehlo::GatherDimensionNumbers(
    offset_dims = offset_axes - 1L,
    collapsed_slice_dims = collapsed_slice_axes - 1L,
    operand_batching_dims = x_batching_axes - 1L,
    start_indices_batching_dims = start_indices_batching_axes - 1L,
    start_index_map = start_index_map - 1L,
    index_vector_dim = index_vector_axis - 1L
  )

  one <- hlo_tensor(1L, dtype = dtype(start_indices), shape = shape(start_indices))
  start_indices_0based <- hlo_subtract(start_indices, one)

  result <- hlo_gather(
    x,
    start_indices_0based,
    gather_dimension_numbers = gdn_0based,
    slice_sizes = slice_sizes,
    indices_are_sorted = indices_are_sorted
  )

  list(result)
}

prim_chol[["stablehlo"]] <- function(x, lower) {
  L <- hlo_cholesky(x, lower = lower)
  # The non-triangular part of the output is implementation-defined.
  # Zero it out so downstream code (including reverse rules) never sees garbage.
  op_shape <- shape(x$value_type)
  n <- op_shape[length(op_shape)]
  mat_shape <- c(n, n)
  rows <- hlo_iota(iota_dimension = 0L, dtype = "i32", shape = mat_shape, func = x$func)
  cols <- hlo_iota(iota_dimension = 1L, dtype = "i32", shape = mat_shape, func = x$func)
  mask <- if (lower) {
    hlo_compare(rows, cols, comparison_direction = "GE", compare_type = "SIGNED")
  } else {
    hlo_compare(rows, cols, comparison_direction = "LE", compare_type = "SIGNED")
  }
  if (length(op_shape) > 2L) {
    mask <- hlo_broadcast_in_dim(mask, (length(op_shape) - 2L):(length(op_shape) - 1L), op_shape)
  }
  zero <- hlo_tensor(0L, dtype = dtype(x), shape = op_shape)
  list(hlo_select(mask, L, zero))
}

prim_triangular_solve[["stablehlo"]] <- function(a, b, left_side, lower, unit_diagonal, transpose_a, output_types) {
  list(hlo_triangular_solve(
    a,
    b,
    left_side = left_side,
    lower = lower,
    unit_diagonal = unit_diagonal,
    transpose_a = if (transpose_a) "TRANSPOSE" else "NO_TRANSPOSE",
    output_types = output_types
  ))
}

prim_qr[["stablehlo"]] <- function(x) {
  vt <- x$value_type
  tt <- vt$type
  dt <- tt$dtype
  shp <- shape(tt)
  m <- shp[1L]
  n <- shp[2L]
  k <- min(m, n)

  # pjrt's QR is split across two custom calls (matching jaxlib + LAPACK):
  #   geqrf(A) -> packed (m, n), tau (k)
  #     where R is encoded in the upper triangle of the first k rows of packed
  #     and the Householder reflectors live in the strict lower triangle.
  #   orgqr(packed, tau) -> Q (m, k)
  # We compose them here and reconstruct R from packed via slice + triu.
  packed_type <- vt(dtype = dt, shape = c(m, n))
  tau_type <- vt(dtype = dt, shape = k)

  geqrf_out <- hlo_custom_call(
    x,
    call_target_name = "geqrf",
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = list(packed_type, tau_type),
    operand_layouts = col_major_layouts(2L),
    result_layouts = col_major_layouts(2L, 1L)
  )
  packed <- geqrf_out[[1L]]
  tau <- geqrf_out[[2L]]

  q_type <- vt(dtype = dt, shape = c(m, k))
  Q <- hlo_custom_call(
    packed,
    tau,
    call_target_name = "orgqr",
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = list(q_type),
    operand_layouts = col_major_layouts(2L, 1L),
    result_layouts = col_major_layouts(2L)
  )

  # R is the upper triangle of the first k rows of packed. Slice if m > k
  # (= m > n); when m <= n, k = m and packed already has the right row count.
  packed_top <- if (m > k) {
    hlo_slice(
      packed,
      start_indices = c(0L, 0L),
      limit_indices = c(k, n),
      strides = c(1L, 1L)
    )
  } else {
    packed
  }
  r_shape <- c(k, n)
  rows <- hlo_iota(iota_dimension = 0L, dtype = "i32", shape = r_shape, func = x$func)
  cols <- hlo_iota(iota_dimension = 1L, dtype = "i32", shape = r_shape, func = x$func)
  triu_mask <- hlo_compare(rows, cols, comparison_direction = "LE", compare_type = "SIGNED")
  zero <- hlo_tensor(0L, dtype = dt, shape = r_shape)
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

  iota0 <- hlo_iota(iota_dimension = 0L, dtype = "i32", shape = n, func = func)
  one_n <- hlo_tensor(1L, dtype = "i32", shape = n, func = func)
  init_perm <- hlo_add(iota0, one_n)
  init_i <- hlo_scalar(0L, dtype = "i32", func = func)

  cond_func <- stablehlo::local_func("")
  i <- hlo_input("i", "i32")
  hlo_input("perm", "i32", n) # unused in cond; declared to match state shape
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
  i <- hlo_input("i", "i32")
  perm <- hlo_input("perm", "i32", n)
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

prim_lu[["stablehlo"]] <- function(x) {
  vt <- x$value_type
  tt <- vt$type
  dt <- tt$dtype
  shp <- shape(tt)
  m <- shp[1L]
  n <- shp[2L]
  k <- min(m, n)

  lu_type <- vt(dtype = dt, shape = c(m, n))
  piv_type <- vt(dtype = "i32", shape = k)

  out <- hlo_custom_call(
    x,
    call_target_name = "lu",
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = list(lu_type, piv_type),
    operand_layouts = col_major_layouts(2L),
    result_layouts = col_major_layouts(2L, 1L)
  )
  LU <- out[[1L]]
  pivots <- out[[2L]]
  permutation <- pivots_to_permutation(pivots, m)
  list(LU, pivots, permutation)
}

prim_svd[["stablehlo"]] <- function(x) {
  tt <- x$value_type$type
  dt <- tt$dtype
  shp <- shape(tt)
  m <- shp[1L]
  n <- shp[2L]
  k <- min(m, n)

  u_type <- vt(dtype = dt, shape = c(m, k))
  s_type <- vt(dtype = dt, shape = k)
  vt_type <- vt(dtype = dt, shape = c(k, n))

  # cuSOLVER's gesvd requires m >= n. When m < n on CUDA we mirror JAX's
  # layout-flip trick: declare x / U / Vt as row-major instead of
  # column-major and tell the FFI handler `transposed = TRUE`. The
  # col-major (m, n) buffer is bit-identical to a row-major (n, m) view of
  # A^T, so cuSOLVER (with m and n swapped) computes A^T = V S U^T and
  # writes its U / Vt outputs straight into the original Vt / U slots —
  # also bit-identical under the col-major <-> row-major reinterpretation.
  # No explicit transpose. The CUDA handler always reads a `transposed`
  # attribute, so we emit it (with both values) whenever targeting CUDA;
  # other platforms (host LAPACK gesdd handles any shape) take the plain
  # path with no backend_config.
  on_cuda <- identical(current_platform(), "cuda")
  transposed <- on_cuda && m < n

  if (on_cuda) {
    backend_config <- stablehlo::CustomOpBackendConfig(list(
      stablehlo::BoolAttr(name = "transposed", value = transposed)
    ))
    if (transposed) {
      row_major_2 <- c(1L, 0L)
      row_major_1 <- 0L
      operand_layouts <- list(row_major_2)
      result_layouts <- list(row_major_2, row_major_1, row_major_2)
    } else {
      operand_layouts <- col_major_layouts(2L)
      result_layouts <- col_major_layouts(2L, 1L, 2L)
    }
    out <- hlo_custom_call(
      x,
      call_target_name = "svd",
      api_version = 4L,
      has_side_effect = FALSE,
      backend_config = backend_config,
      output_types = list(u_type, s_type, vt_type),
      operand_layouts = operand_layouts,
      result_layouts = result_layouts
    )
  } else {
    out <- hlo_custom_call(
      x,
      call_target_name = "svd",
      api_version = 4L,
      has_side_effect = FALSE,
      output_types = list(u_type, s_type, vt_type),
      operand_layouts = col_major_layouts(2L),
      result_layouts = col_major_layouts(2L, 1L, 2L)
    )
  }
  list(out[[2L]], out[[1L]], out[[3L]])
}

prim_eigh[["stablehlo"]] <- function(x) {
  tt <- x$value_type$type
  dt <- tt$dtype
  n <- shape(tt)[1L]

  v_type <- vt(dtype = dt, shape = c(n, n))
  w_type <- vt(dtype = dt, shape = n)

  # The "eigh" custom call returns (vectors, values) positionally. We
  # re-bundle to (values, vectors) matching `base::eigen()`.
  out <- hlo_custom_call(
    x,
    call_target_name = "eigh",
    api_version = 4L,
    has_side_effect = FALSE,
    output_types = list(v_type, w_type),
    operand_layouts = col_major_layouts(2L),
    result_layouts = col_major_layouts(2L, 1L)
  )
  list(out[[2L]], out[[1L]]) # values, vectors
}

prim_convolution[["stablehlo"]] <- function(
  x,
  kernel,
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
  x_dilation,
  kernel_dilation,
  feature_group_count,
  batch_group_count,
  precision
) {
  shlo_dn <- stablehlo::ConvDimensionNumbers(
    input_batch_dimension = input_batch_axis - 1L,
    input_feature_dimension = input_feature_axis - 1L,
    input_spatial_dimensions = input_spatial_axes - 1L,
    kernel_input_feature_dimension = kernel_input_feature_axis - 1L,
    kernel_output_feature_dimension = kernel_output_feature_axis - 1L,
    kernel_spatial_dimensions = kernel_spatial_axes - 1L,
    output_batch_dimension = output_batch_axis - 1L,
    output_feature_dimension = output_feature_axis - 1L,
    output_spatial_dimensions = output_spatial_axes - 1L
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
