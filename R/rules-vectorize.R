#' @include primitives.R
#' @include vectorize.R
NULL

# Trivially vectorizable primitives where appending a batch axis does not change the computation.
# The linear algebra primitives take a batch of matrices, whose batch axes trail them.
local({
  for (prim in c(
    primitives_tagged(c("elementwise", "reduction", "cumulative")),
    list(prim_rev, prim_concatenate, prim_bitcast_convert, prim_top_k),
    list(prim_chol, prim_triangular_solve, prim_qr, prim_lu, prim_svd, prim_eigh)
  )) {
    prim[["vectorize"]] <- rule_vectorize()
  }
})

prim_transpose[["vectorize"]] <- rule_vectorize(params = list(perm = vec_axis_map()))
prim_reshape[["vectorize"]] <- rule_vectorize(params = list(shape = vec_shape()))
prim_static_slice[["vectorize"]] <- rule_vectorize(
  params = list(
    start_indices = vec_per_axis(1L),
    end_indices = vec_per_axis(function(size) size),
    strides = vec_per_axis(1L)
  )
)
prim_pad[["vectorize"]] <- rule_vectorize(
  params = list(
    edge_padding_low = vec_per_axis(0L),
    edge_padding_high = vec_per_axis(0L),
    interior_padding = vec_per_axis(0L)
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

# The barrier passes its operands through, so every one of them gets the batch
# axis, as every output of a call with a batched input has it.
prim_optimization_barrier[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  unname(prim_optimization_barrier(Map(batch_operand, inputs, batched, MoreArgs = list(size = size))))
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

# The output of `gather` has an axis for every axis of `start_indices` but the
# index vector axis, in their order, and puts the offset axes in between. The
# batch axis of `start_indices` alone is one more such axis, the last; an index
# vector axis that is implicit, after the last axis, moves behind it. The batch
# axis of `x` alone is one more offset axis, which the slices span whole and
# the output gets last. When both are batched, their batch axes are a pair of
# batching axes, which match the slices of `x` with the indices into them.
prim_gather[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  x <- inputs[[1L]]
  start_indices <- inputs[[2L]]
  n_indices <- naxes(start_indices)
  if (batched[[2L]] && params$index_vector_axis == n_indices) {
    params$index_vector_axis <- n_indices + 1L
  }
  if (all(batched)) {
    params$x_batching_axes <- c(params$x_batching_axes, naxes(x))
    params$start_indices_batching_axes <- c(params$start_indices_batching_axes, n_indices)
    params$slice_sizes <- c(params$slice_sizes, 1L)
  } else if (batched[[1L]]) {
    n_out <- n_indices - (params$index_vector_axis <= n_indices) + length(params$offset_axes)
    params$offset_axes <- c(params$offset_axes, n_out + 1L)
    params$slice_sizes <- c(params$slice_sizes, size)
  }
  list(do.call(prim_gather, c(list(x, start_indices), params)))
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
