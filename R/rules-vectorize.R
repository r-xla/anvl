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
# batching axes, which match the slices of `x` with the indices into them; a
# batching axis takes a slice of size 1, which an empty batch does not have, so
# then the gather goes slice by slice.
#
# Indices sorted within one slice need not be across the batch. Unique ones
# stay unique where the batching axes tell the slices apart, but with only the
# indices batched, the indices of all slices go into the same `x`.
prim_gather[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  if (all(batched) && size == 0L) {
    gather <- function(operands) list(do.call(prim_gather, c(operands, params)))
    return(vectorize_by_loop(gather, inputs, batched, size))
  }
  x <- inputs[[1L]]
  start_indices <- inputs[[2L]]
  n_indices <- naxes(start_indices)
  if (batched[[2L]]) {
    params$indices_are_sorted <- FALSE
    if (params$index_vector_axis == n_indices) {
      params$index_vector_axis <- n_indices + 1L
    }
  }
  if (batched[[2L]] && !batched[[1L]]) {
    params$unique_indices <- FALSE
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

# Every slice of the output is a slice of `x` with the updates of its own slice,
# so `x` and `update` both get the batch axis. The batch axis of
# `scatter_indices` makes it a pair of batching axes of `x` and the indices,
# which match the slices of `x` with the indices into them, and the last scatter
# axis of `update`; an index vector axis that is implicit, after the last axis,
# moves behind it. Indices sorted within one slice need not be across the batch,
# but unique ones stay unique, as the batching axes tell the slices apart.
# Without it, the batch axis of `x` is one more window axis, the last, which the
# last axis of `update` spans whole. The update computation closes over no
# traced value, so it is the same for every slice.
prim_scatter[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  x <- batch_operand(inputs[[1L]], batched[[1L]], size)
  scatter_indices <- inputs[[2L]]
  update <- batch_operand(inputs[[3L]], batched[[3L]], size)
  if (batched[[2L]]) {
    n_indices <- naxes(scatter_indices)
    if (params$index_vector_axis == n_indices) {
      params$index_vector_axis <- n_indices + 1L
    }
    params$x_batching_axes <- c(params$x_batching_axes, naxes(x))
    params$scatter_indices_batching_axes <- c(params$scatter_indices_batching_axes, n_indices)
    params$indices_are_sorted <- FALSE
  } else {
    params$update_window_axes <- c(params$update_window_axes, naxes(update))
  }
  graph_desc_add(
    prim_scatter,
    c(list(x, scatter_indices, update), inputs[-(1:3)]),
    params = params,
    infer_fn = infer_scatter
  )
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

# The fork of a batch of random states forks each of them (see
# `infer_fork_rng_state()`).
prim_fork_rng_state[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  operands <- Map(batch_operand, inputs, batched, MoreArgs = list(size = size))
  graph_desc_add(prim_fork_rng_state, operands, infer_fn = infer_fork_rng_state)
})

# The reducer works on scalars, so it stays as it is, and the batch axis of `x`
# is one more axis the reduction keeps. An initial value that is mapped over is
# not a scalar any more: then the reduction goes slice by slice.
prim_reduce[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  reduce <- function(operands) graph_desc_add(prim_reduce, operands, params = params, infer_fn = infer_reduce)
  if (any(batched[-1L])) {
    return(vectorize_by_loop(reduce, inputs, batched, size))
  }
  reduce(inputs)
})

# The predicate `pred` of every slice, for the values `yes` and `no` of a
# batch: `prim_ifelse()` with `pred` broadcast along the batch axis.
select_slices <- function(pred, yes, no) {
  prim_ifelse(prim_broadcast_in_axes(pred, shape(yes), naxes(yes)), yes, no)
}

# With a predicate that is not mapped over, both branches are vectorized. With
# one that is, each slice may take another branch: each branch that some slice
# takes runs for all of them, and each slice selects the result of its own.
#
# A slice that does not take a branch still runs it, and its result there is
# discarded, but its derivative is not: `0 * Inf` from, e.g., a `sqrt()` the
# slice guards against would reach the gradient as `NaN`. So a branch runs a
# slice that does not take it on the mapped inputs of one that does, slice `k`:
# it reads each of them through `ifelse(pred, x, x[k])`. The branch is valid
# there, as slice `k` takes it. The reverse rule of `prim_ifelse()` gives the
# slices that do not take the branch no cotangent, and `x[k]` none either, as it
# passes `stop_gradient()`. A branch that no slice takes has no slice `k` and is
# not run at all. Inputs that are not mapped over stay as they are, so a
# weight matrix the branches share is not copied for every slice.
prim_if[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  pred <- inputs[[1L]]
  captures <- inputs[-1L]
  captures_batched <- batched[-1L]
  replay <- function(graph, captures) {
    graph_vectorize_as(graph, captures, captures_batched, size, rep(TRUE, length(graph$outputs)))
  }
  if (!batched[[1L]]) {
    return(prim_if(pred, function() replay(params$true, captures), function() replay(params$false, captures)))
  }
  if (size == 0L) {
    return(lapply(params$true$outputs, \(out) zeros(dtype(out$aval), c(shape(out$aval), 0L))))
  }
  # The branch `graph`, taken by the slices `takes` flags.
  run <- function(graph, takes) {
    taken <- function() {
      k <- prim_reshape(prim_which_max(takes, axis = 1L), 1L)
      borrowed <- captures
      borrowed[captures_batched] <- lapply(captures[captures_batched], function(x) {
        n <- naxes(x)
        # A gather rather than a dynamic slice, so that an enclosing
        # `vectorize()` can map over `k` too.
        x_k <- prim_gather(
          x,
          k,
          slice_sizes = c(shape(x)[-n], 1L),
          offset_axes = seq_len(n - 1L),
          collapsed_slice_axes = n,
          x_batching_axes = integer(),
          start_indices_batching_axes = integer(),
          start_index_map = n,
          index_vector_axis = 1L
        )
        if (is_dtype_float(dtype(x_k))) {
          x_k <- stop_gradient(x_k)
        }
        select_slices(takes, x, prim_broadcast_in_axes(x_k, shape(x), seq_len(n - 1L)))
      })
      replay(graph, borrowed)
    }
    none <- function() lapply(graph$outputs, \(out) zeros(dtype(out$aval), c(shape(out$aval), size)))
    prim_if(prim_any(takes, axes = 1L), taken, none)
  }
  Map(select_slices, list(pred), run(params$true, pred), run(params$false, prim_not(pred)))
})

# The values the loop carries are batched where they enter batched or the body
# makes them so, and stay as they are otherwise, so that, e.g., a counter keeps
# the condition the same for all slices. With a condition that is mapped over,
# the loop runs while it holds for any slice, and a slice for which it no longer
# holds keeps its state, so all of the state is batched. The condition of every
# slice is carried along with the state, so that it is evaluated once per step.
prim_while[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  is_state <- seq_len(length(inputs) - params$n_captures)
  captures <- inputs[-is_state]
  captures_batched <- batched[-is_state]
  state_batched <- carry_batched(params$body, batched[is_state], captures_batched)
  pred_batched <- graph_batched_outputs(params$cond, c(state_batched, captures_batched))
  if (pred_batched) {
    state_batched[] <- TRUE
  }
  in_batched <- c(state_batched, captures_batched)
  pred_of <- function(state) {
    graph_vectorize_as(params$cond, c(unname(state), captures), in_batched, size, pred_batched)[[1L]]
  }
  body_of <- function(state) {
    stats::setNames(
      graph_vectorize_as(params$body, c(unname(state), captures), in_batched, size, state_batched),
      names(state)
    )
  }
  init <- Map(
    function(x, is_batched, want) if (want) batch_operand(x, is_batched, size) else x,
    inputs[is_state],
    batched[is_state],
    state_batched
  )
  names(init) <- paste0("s", is_state)

  out <- if (pred_batched) {
    # FIXME: the body runs on every slice, also on one whose condition is false
    # (done, or never entered the loop), and only then do those slices keep
    # their old state. The body may rely on the condition, e.g. an inner loop
    # that only terminates for the states it admits, which then never ends for
    # such a slice and hangs the whole loop (`jax.vmap()` of a `while_loop()`
    # does the same). Run those slices on the state and the mapped captures of
    # a slice whose condition holds, as the `prim_if` rule does for a slice
    # that does not take a branch.
    # `pred` is the condition of every slice for the state the step starts from.
    init$pred <- pred_of(init)
    prim_while(
      init,
      function(...) prim_any(list(...)$pred, axes = 1L),
      function(...) {
        state <- list(...)
        old <- state[names(state) != "pred"]
        new <- Map(function(n, o) select_slices(state$pred, n, o), body_of(old), old)
        c(new, list(pred = pred_of(new)))
      }
    )[is_state]
  } else {
    prim_while(init, function(...) pred_of(list(...)), function(...) body_of(list(...)))
  }
  Map(batch_operand, unname(out), state_batched, MoreArgs = list(size = size))
})

# As in `prim_while()`, the carry is batched where it enters batched or the body
# makes it so, and so is a stacked output where the body makes it depend on what
# is batched. The loop runs over the last axis of `xs`, which a batched one has
# before its batch axis: the two swap places on the way in, and on the batched
# stacked outputs on the way out.
prim_scan[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  is_carry <- seq_len(params$n_carry)
  is_xs <- params$n_carry + seq_len(params$n_xs)
  captures <- inputs[-c(is_carry, is_xs)]
  other_batched <- batched[-is_carry]
  carry_b <- carry_batched(params$body, batched[is_carry], other_batched)
  out_b <- graph_batched_outputs(params$body, c(carry_b, other_batched))[-is_carry]

  init <- Map(
    function(x, is_batched, want) if (want) batch_operand(x, is_batched, size) else x,
    inputs[is_carry],
    batched[is_carry],
    carry_b
  )
  xs <- Map(
    function(x, is_batched) if (is_batched) move_axis(x, naxes(x), naxes(x) - 1L) else x,
    inputs[is_xs],
    batched[is_xs]
  )
  body <- function(carry, x) {
    out <- graph_vectorize_as(params$body, c(carry, x, captures), c(carry_b, other_batched), size, c(carry_b, out_b))
    list(carry = out[is_carry], out = out[-is_carry])
  }
  out <- prim_scan(init, xs, body, steps = params$steps, reverse = params$reverse)
  ys <- Map(
    function(y, is_batched) if (is_batched) move_axis(y, naxes(y) - 1L, naxes(y)) else add_batch_axis(y, size),
    out$out,
    out_b
  )
  c(Map(batch_operand, out$carry, carry_b, MoreArgs = list(size = size)), ys)
})

# A call on the batch: `fn` replayed with the batch axis, and `forward` and
# `rule` mapped over the slices with `vectorize()`, which gives each slice its
# own `out`, `grad` and residuals and passes an argument that is not mapped over
# as it is. The gradient of such an argument is the sum of those of the slices.
prim_custom_gradient[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  fn <- params$fn
  arg_idx <- seq_len(length(inputs) - params$n_captures)
  captures <- inputs[-arg_idx]
  args <- unflatten(fn$in_tree, inputs[arg_idx])
  arg_names <- names(args)
  # by entry, whether each array is mapped over, and the `in_axes` that says so
  mapped <- unflatten(fn$in_tree, as.list(batched[arg_idx]))
  as_axes <- function(m) {
    if (is.list(m)) {
      lapply(m, as_axes)
    } else if (m) {
      -1L
    }
  }
  in_axes <- lapply(mapped, as_axes)
  any_mapped <- any(batched[arg_idx])

  fn_batched <- function(...) {
    entries <- list(...)[arg_names]
    outs <- graph_vectorize_as(fn, c(flatten(entries), captures), batched, size, rep(TRUE, length(fn$outputs)))
    unflatten(fn$out_tree, outs)
  }

  forward <- params$forward
  forward_batched <- if (!is.null(forward)) {
    function(...) {
      a <- list(...)
      one <- function(entries) do.call(forward, c(entries, list(required = a$required)))
      if (!any_mapped) {
        # every slice computes the same, which only needs the batch axis
        res <- one(a[arg_names])
        return(map_tree(res, \(x) batch_operand(as_anvl_array(x), FALSE, size)))
      }
      vectorize(one, in_axes = list(entries = in_axes))(a[arg_names])
    }
  }

  rule <- params$rule
  rule_batched <- function(...) {
    a <- list(...)
    required <- a$required
    one <- function(entries, out, grad, residuals = NULL) {
      extra <- if (!is.null(forward)) list(residuals = residuals)
      grads <- do.call(rule, c(entries, list(out = out, grad = grad, required = required), extra))
      grads[intersect(names(required)[required], names(grads))]
    }
    axes <- list(entries = in_axes, out = -1L, grad = -1L)
    call_args <- list(entries = a[arg_names], out = a$out, grad = a$grad)
    if (!is.null(forward)) {
      call_args$residuals <- a$residuals
      if (length(flatten(a$residuals))) {
        axes$residuals <- -1L
      }
    }
    grads <- do.call(vectorize(one, in_axes = axes), call_args)
    # a slice's gradient of an array it shares with the others adds to theirs
    stats::setNames(
      lapply(names(grads), function(nm) {
        g <- grads[[nm]]
        leaves <- Map(
          function(leaf, is_mapped) if (is_mapped) leaf else nv_sum(leaf, axes = naxes(leaf)),
          flatten(g),
          flatten(mapped[[nm]])
        )
        unflatten(build_tree(g), leaves)
      }),
      names(grads)
    )
  }

  flatten(prim_custom_gradient(args, fn = fn_batched, rule = rule_batched, forward = forward_batched))
})

# The batch axis is folded into an axis of the operands whose slices
# `prim_convolution()` keeps apart: the batch axis of `x` when only `x` is
# mapped over, the output feature axis of `kernel` when only `kernel` is, and
# both feature axes -- one feature group per slice -- when both are. Where
# groups of the convolution's own are in the way, and for an empty batch, it
# goes slice by slice.
prim_convolution[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
  x <- inputs[[1L]]
  kernel <- inputs[[2L]]
  only_x <- batched[[1L]] && !batched[[2L]]
  if (size == 0L || params$batch_group_count != 1L || (!only_x && params$feature_group_count != 1L)) {
    convolve <- function(operands) list(do.call(prim_convolution, c(operands, params)))
    return(vectorize_by_loop(convolve, inputs, batched, size))
  }
  if (only_x) {
    x <- merge_batch_axis(x, params$x_batch_axis)
    out_axis <- params$output_batch_axis
  } else {
    kernel <- merge_batch_axis(kernel, params$kernel_output_feature_axis)
    out_axis <- params$output_feature_axis
    if (batched[[1L]]) {
      x <- merge_batch_axis(x, params$x_feature_axis)
      params$feature_group_count <- size
    }
  }
  split_batch_axis(do.call(prim_convolution, c(list(x, kernel), params)), out_axis, size)
})
