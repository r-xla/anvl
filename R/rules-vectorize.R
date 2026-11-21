#' @include primitives.R
#' @include vectorize.R
NULL

# Primitives that work on an extra trailing axis as they are: the element-wise
# ones, and those whose parameters name axes, which keep their positions in
# front of the batch axis. anvl's bitcast packs from, or unpacks into, a
# leading axis, and `prim_top_k()` works along the first axis with the trailing
# ones as batch axes, which leaves the batch axis last. A scalar operand of an
# element-wise primitive is broadcast only if it is mapped over (see
# `batch_operands()`).
for (prim in c(
  primitives_tagged(c("elementwise", "reduction", "cumulative")),
  list(prim_rev, prim_concatenate, prim_bitcast_convert, prim_top_k)
)) {
  prim[["vectorize"]] <- rule_vectorize()
}
rm(prim)

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
# `dot_general` places the free axes of `lhs` first, then those of `rhs`, then
# the batching axes, so a batching axis appended last is the last axis of the
# output, and a free batch axis is moved there.
prim_dot_general[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  lhs <- inputs[[1L]]
  rhs <- inputs[[2L]]
  contracting <- params$contracting_axes
  batching <- params$batching_axes
  n_batching <- length(batching[[1L]])
  if (all(batched)) {
    batching <- list(c(batching[[1L]], naxes(lhs)), c(batching[[2L]], naxes(rhs)))
  }
  out <- prim_dot_general(
    lhs,
    rhs,
    contracting_axes = contracting,
    batching_axes = batching,
    precision = params$precision
  )
  if (all(batched)) {
    return(list(out))
  }
  from <- if (batched[[1L]]) {
    naxes(lhs) - length(contracting[[1L]]) - n_batching
  } else {
    naxes(out) - n_batching
  }
  list(move_axis(out, from, naxes(out)))
})

# A draw from a batch of random states draws from each of them (see
# `infer_sample_bits()`): the batch axis trails the key and the counter, and
# trails the draws.
prim_sample_bits[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  operands <- Map(batch_operand, inputs, batched, MoreArgs = list(size = size))
  graph_desc_add(prim_sample_bits, operands, params = params, infer_fn = infer_sample_bits)
})
