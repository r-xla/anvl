#' @include primitives.R
#' @include vectorize.R
NULL

# Primitives that work on an extra trailing axis as they are: the element-wise
# ones, and those whose parameters name axes, which keep their positions in
# front of the batch axis. anvl's bitcast packs from, or unpacks into, a
# leading axis, which leaves the batch axis last.
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
  prim_sum,
  prim_prod,
  prim_max,
  prim_min,
  prim_any,
  prim_all,
  prim_rev,
  prim_cumsum,
  prim_cumprod,
  prim_cummax,
  prim_cummin,
  prim_which_max,
  prim_which_min,
  prim_concatenate
)) {
  prim[["vectorize"]] <- rule_vectorize()
}
rm(prim)

prim_clamp[["vectorize"]] <- rule_vectorize(scalar = 2:3)
prim_ifelse[["vectorize"]] <- rule_vectorize(scalar = 1L)

prim_transpose[["vectorize"]] <- rule_vectorize(params = list(perm = param_axis_map()))
prim_reshape[["vectorize"]] <- rule_vectorize(params = list(shape = param_shape()))
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

# The batch axis of the operand becomes the last axis of the output, after the
# axes of `shape`.
prim_broadcast_in_axes[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  prim_broadcast_in_axes(
    inputs[[1L]],
    shape = c(params$shape, size),
    broadcast_axes = c(params$broadcast_axes, length(params$shape) + 1L)
  )
})

prim_print[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  list(prim_print(inputs[[1L]]))
})

prim_sort[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  prim_sort(
    batch_operands(inputs, batched, size),
    axis = params$axis,
    decreasing = params$decreasing,
    stable = params$stable
  )
})

# `prim_top_k()` works along the last axis, so the batch axis moves in front for
# the call, and back to the end of every output after it.
prim_top_k[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  x <- inputs[[1L]]
  n <- naxes(x)
  out <- prim_top_k(move_axis(x, n, 1L), k = params$k, indices = params$indices)
  lapply(out, move_axis, from = 1L, to = n)
})

# The start indices are scalars: the slice starts at the beginning of the batch
# axis and spans all of it.
prim_dynamic_slice[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  if (any(batched[-1L])) {
    cli_abort("{.fn vectorize} cannot map over the start indices of {.fn prim_dynamic_slice} yet.")
  }
  list(do.call(
    prim_dynamic_slice,
    c(inputs, list(1L), list(slice_sizes = c(params$slice_sizes, size)))
  ))
})

prim_dynamic_update_slice[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  if (any(batched[-(1:2)])) {
    cli_abort("{.fn vectorize} cannot map over the start indices of {.fn prim_dynamic_update_slice} yet.")
  }
  operands <- batch_operands(inputs[1:2], batched[1:2], size)
  list(do.call(prim_dynamic_update_slice, c(operands, inputs[-(1:2)], list(1L))))
})

# The batch axis of a batched operand becomes a batching axis when both are
# batched, and a free axis otherwise: the last free axis of `lhs` or of `rhs`.
# `dot_general` places the batching axes first, then the free axes of `lhs`,
# then those of `rhs`, so the batch axis is moved to the end unless it is the
# last free axis of `rhs` already.
prim_dot_general[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  contracting <- params$contracting_axes
  batching <- params$batching_axes
  n_batching <- length(batching[[1L]])
  from <- if (all(batched)) {
    batching <- list(c(batching[[1L]], naxes(lhs)), c(batching[[2L]], naxes(rhs)))
    n_batching + 1L
  } else if (batched[[1L]]) {
    naxes(lhs) - length(contracting[[1L]])
  }
  out <- prim_dot_general(
    lhs,
    rhs,
    contracting_axes = contracting,
    batching_axes = batching,
    precision = params$precision
  )
  if (is.null(from)) list(out) else list(move_axis(out, from, naxes(out)))
})

# A draw from a batch of random states draws from each of them (see
# `infer_sample_bits()`): the batch axis trails the key and the counter, and
# trails the draws.
prim_sample_bits[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  operands <- Map(batch_operand, inputs, batched, MoreArgs = list(size = size))
  graph_desc_add(prim_sample_bits, operands, params = params, infer_fn = infer_sample_bits)
})
