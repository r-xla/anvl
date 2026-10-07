#' @include primitives.R
#' @include vectorize.R
NULL

# Primitives that work on an extra leading axis as they are, and whose
# parameters do not refer to the axes of their operands: the element-wise ones,
# and those that only look at their trailing axes.
for (prim in list(
  prim_add,
  prim_mul,
  prim_sub,
  prim_negate,
  prim_div,
  prim_pow,
  prim_eq,
  prim_ne,
  prim_gt,
  prim_ge,
  prim_lt,
  prim_le,
  prim_pmax,
  prim_pmin,
  prim_remainder,
  prim_and,
  prim_not,
  prim_or,
  prim_xor,
  prim_shift_left,
  prim_shift_right_logical,
  prim_shift_right_arithmetic,
  prim_atan2,
  prim_abs,
  prim_sqrt,
  prim_rsqrt,
  prim_log,
  prim_tanh,
  prim_tan,
  prim_sin,
  prim_cos,
  prim_floor,
  prim_ceiling,
  prim_sign,
  prim_exp,
  prim_expm1,
  prim_log1p,
  prim_cbrt,
  prim_plogis,
  prim_acos,
  prim_acosh,
  prim_asin,
  prim_asinh,
  prim_atan,
  prim_atanh,
  prim_cosh,
  prim_sinh,
  prim_digamma,
  prim_lgamma,
  prim_psigamma,
  prim_erf,
  prim_erf_inv,
  prim_erfc,
  prim_is_finite,
  prim_popcnt,
  prim_round,
  prim_convert,
  prim_bitcast_convert,
  prim_top_k,
  prim_chol,
  prim_triangular_solve
)) {
  prim[["vectorize"]] <- rule_vectorize()
}
rm(prim)

prim_clamp[["vectorize"]] <- rule_vectorize(scalar = 2:3)
prim_ifelse[["vectorize"]] <- rule_vectorize(scalar = 1L)

for (prim in list(prim_sum, prim_prod, prim_max, prim_min, prim_any, prim_all, prim_rev)) {
  prim[["vectorize"]] <- rule_vectorize(params = list(axes = param_axes()))
}
for (prim in list(
  prim_cumsum,
  prim_cumprod,
  prim_cummax,
  prim_cummin,
  prim_which_max,
  prim_which_min,
  prim_concatenate
)) {
  prim[["vectorize"]] <- rule_vectorize(params = list(axis = param_axes()))
}
rm(prim)

prim_transpose[["vectorize"]] <- rule_vectorize(params = list(perm = param_axis_map()))
prim_reshape[["vectorize"]] <- rule_vectorize(params = list(shape = param_shape()))
prim_broadcast_in_axes[["vectorize"]] <- rule_vectorize(
  params = list(shape = param_shape(), broadcast_axes = param_axis_map())
)
prim_static_slice[["vectorize"]] <- rule_vectorize(
  params = list(
    start_indices = param_per_axis(1L),
    end_indices = param_per_axis(function(size) size),
    strides = param_per_axis(1L)
  )
)
prim_pad[["vectorize"]] <- rule_vectorize(
  params = list(
    edge_padding_low = param_per_axis(0L),
    edge_padding_high = param_per_axis(0L),
    interior_padding = param_per_axis(0L)
  ),
  unbatched = 2L
)

prim_print[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  list(prim_print(inputs[[1L]]))
})

prim_sort[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  prim_sort(
    batch_operands(inputs, batched, size),
    axis = params$axis + 1L,
    decreasing = params$decreasing,
    stable = params$stable
  )
})

# The start indices are scalars: the slice starts at the beginning of the batch
# axis and spans all of it.
prim_dynamic_slice[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  if (any(batched[-1L])) {
    cli_abort("{.fn vectorize} cannot map over the start indices of {.fn prim_dynamic_slice} yet.")
  }
  list(do.call(
    prim_dynamic_slice,
    c(inputs[1L], list(1L), inputs[-1L], list(slice_sizes = c(size, params$slice_sizes)))
  ))
})

prim_dynamic_update_slice[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  if (any(batched[-(1:2)])) {
    cli_abort("{.fn vectorize} cannot map over the start indices of {.fn prim_dynamic_update_slice} yet.")
  }
  operands <- batch_operands(inputs[1:2], batched[1:2], size)
  list(do.call(prim_dynamic_update_slice, c(operands, list(1L), inputs[-(1:2)])))
})

# The batch axis of a batched operand becomes a batching axis when both are
# batched, and a free axis otherwise, which `dot_general` places after the
# batching axes (lhs) or after all axes of lhs (rhs).
prim_dot_general[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  contracting <- params$contracting_axes
  batching <- params$batching_axes
  shift <- function(axes, b) if (b) axes + 1L else axes
  contracting <- Map(shift, contracting, batched)
  n_batching <- length(batching[[1L]])
  if (all(batched)) {
    batching <- lapply(batching, \(axes) c(1L, axes + 1L))
    from <- 1L
  } else {
    batching <- Map(shift, batching, batched)
    from <- if (batched[[1L]]) {
      n_batching + 1L
    } else {
      naxes(lhs) - length(contracting[[1L]]) + 1L
    }
  }
  out <- prim_dot_general(
    lhs,
    rhs,
    contracting_axes = contracting,
    batching_axes = batching,
    precision = params$precision
  )
  list(move_axis(out, from, 1L))
})

# The batch axis is a batching axis of both operands when both are batched. When
# only `x` is, it is a window axis spanning the whole batch; when only the
# indices are, it is one more axis of indices. Either way it lands first among
# the output's axes.
prim_gather[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  p <- params
  x <- inputs[[1L]]
  idx <- inputs[[2L]]
  p$collapsed_slice_axes <- p$collapsed_slice_axes + batched[[1L]]
  p$start_index_map <- p$start_index_map + batched[[1L]]
  if (all(batched)) {
    p$x_batching_axes <- c(1L, p$x_batching_axes + 1L)
    p$start_indices_batching_axes <- c(1L, p$start_indices_batching_axes + 1L)
    p$slice_sizes <- c(1L, p$slice_sizes)
    p$offset_axes <- p$offset_axes + 1L
    p$index_vector_axis <- p$index_vector_axis + 1L
  } else if (batched[[1L]]) {
    p$x_batching_axes <- p$x_batching_axes + 1L
    p$slice_sizes <- c(size, p$slice_sizes)
    p$offset_axes <- c(1L, p$offset_axes + 1L)
  } else {
    p$start_indices_batching_axes <- p$start_indices_batching_axes + 1L
    p$offset_axes <- p$offset_axes + 1L
    p$index_vector_axis <- p$index_vector_axis + 1L
  }
  if (batched[[2L]]) {
    # Indices that are sorted or unique within one slice need not be across
    # the batch.
    p$indices_are_sorted <- FALSE
    p$unique_indices <- FALSE
  }
  list(do.call(prim_gather, c(list(x, idx), p)))
})
