#' @include rules-pullback.R
NULL

# Pushforward rules of the elementwise primitives ------------------------------
#
# An element of the result of an elementwise primitive depends only on the
# elements at the same position of its operands, so its derivative with respect
# to each operand is diagonal: its own transpose. The pullback rule multiplies a
# cotangent of the result by that derivative, so given a tangent of an operand
# in place of the cotangent, it gives that tangent's pushforward. The
# pushforward rule of a primitive tagged `"elementwise"` therefore calls its
# pullback rule once per operand with a tangent, and adds up the results.

# The pushforward rule of the elementwise primitive whose pullback rule is
# `pullback`, a `rule_pullback()` without a forward pass.
pushforward_from_pullback <- function(pullback) {
  backward <- pullback$backward
  function(inputs, outputs, tangents, params) {
    out <- outputs[[1L]]
    if (!is_dtype_float(dtype(out))) {
      return(list(NULL))
    }
    # A scalar operand is broadcast to the shape of the result, and the
    # pullback rule sums the cotangent of such an operand back down. Called with
    # every operand at the shape of the result, it sums nothing.
    at_out_shape <- function(x) {
      if (identical(shape(x), shape(out))) x else prim_broadcast_in_axes(x, shape(out), integer())
    }
    operands <- lapply(inputs, at_out_shape)
    with_tangent <- which(!vapply(tangents, is.null, logical(1L)))
    terms <- lapply(with_tangent, function(i) {
      required <- seq_along(inputs) == i
      backward(operands, outputs, list(at_out_shape(tangents[[i]])), params, required)[[i]]
    })
    list(Reduce(prim_add, terms))
  }
}

for (prim in primitives_tagged("elementwise")) {
  if (is.null(prim[["pullback"]]$backward)) {
    cli_abort(
      "Internal error: the elementwise primitive {.field {primitive_def(prim)$name}} has no pullback rule without a forward pass."
    )
  }
  prim[["pushforward"]] <- pushforward_from_pullback(prim[["pullback"]])
}
rm(prim)

# `prim_convert()` is elementwise, but its pullback rule converts back to the
# operand's data type: the tangent of a float result is the operand's tangent
# converted to the result's.
prim_convert[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  out <- outputs[[1L]]
  list(if (is_dtype_float(dtype(out))) prim_convert(tangents[[1L]], dtype(out)))
}

# Pushforward rules of the linear primitives -----------------------------------
#
# A primitive tagged `"linear"` is linear in its float operands jointly, so it
# is its own derivative: the tangent of its result is the primitive applied to
# the tangents of those operands, a zero standing in for a missing one, with
# its other operands -- start indices, which have no tangent -- and its
# parameters unchanged.

# The pushforward rule of the linear primitive `prim`.
pushforward_linear <- function(prim) {
  force(prim)
  function(inputs, outputs, tangents, params) {
    operands <- Map(
      function(x, tangent) tangent %||% if (is_dtype_float(dtype(x))) zeros_like(x) else x,
      inputs,
      tangents
    )
    list(do.call(prim, c(unname(operands), params)))
  }
}

for (prim in primitives_tagged("linear")) {
  prim[["pushforward"]] <- pushforward_linear(prim)
}
rm(prim)

# Printing hands its operand back unchanged, and so its tangent, which it does
# not print.
prim_print[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  list(tangents[[1L]])
}

# The barrier passes its operands through. As the pullback rule does with the
# cotangents, the tangents go behind a barrier of their own.
prim_optimization_barrier[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  given <- !vapply(tangents, is.null, logical(1L))
  out <- vector("list", length(outputs))
  out[given] <- prim_optimization_barrier(tangents[given])
  out
}

# The result of `prim_which_max()` and `prim_which_min()` is an integer, and the
# pullback rule of `prim_bitcast_convert()` gives a zero: none of them passes a
# tangent on.
pushforward_none <- function(inputs, outputs, tangents, params) {
  vector("list", length(outputs))
}

prim_which_max[["pushforward"]] <- pushforward_none
prim_which_min[["pushforward"]] <- pushforward_none
prim_bitcast_convert[["pushforward"]] <- pushforward_none

# Pushforward rules of the multilinear primitives ------------------------------
#
# A primitive that is linear in each of its operands, the others held fixed, is
# its own derivative in each: its tangent is the sum, over the operands with a
# tangent, of the primitive at the operands with that one replaced by its
# tangent.

# The pushforward rule of the multilinear primitive `prim`, whose parameters are
# its arguments of the same names.
pushforward_multilinear <- function(prim) {
  force(prim)
  function(inputs, outputs, tangents, params) {
    with_tangent <- which(!vapply(tangents, is.null, logical(1L)))
    terms <- lapply(with_tangent, function(i) {
      operands <- inputs
      operands[[i]] <- tangents[[i]]
      do.call(prim, c(operands, params))
    })
    list(Reduce(prim_add, terms))
  }
}

for (prim in list(prim_dot_general, prim_convolution)) {
  prim[["pushforward"]] <- pushforward_multilinear(prim)
}
rm(prim)

# Pushforward rules of the reductions ------------------------------------------

# The elements of `x` that attain the extremum share its tangent equally, as
# they share its cotangent in the pullback rule.
prim_max[["pushforward"]] <- prim_min[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  t <- tangents[[1L]]
  sel <- extremum_selection(inputs[[1L]], outputs[[1L]], params$axes, params$drop, dtype(t))
  list(prim_div(prim_sum(prim_mul(t, sel$mask), axes = params$axes, drop = params$drop), sel$count))
}

prim_prod[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  others <- prod_of_others(inputs[[1L]], params$axes)
  list(prim_sum(prim_mul(tangents[[1L]], others), axes = params$axes, drop = params$drop))
}

# Pushforward rules of the cumulative primitives -------------------------------

# The tangent of the running extremum is that of the element it is, at its
# index, where the pullback rule scatters the cotangent to.
prim_cummax[["pushforward"]] <- prim_cummin[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  list(take_along_axis(tangents[[1L]], outputs[[2L]], params$axis), NULL)
}

# Pushforward rules of sorting -------------------------------------------------

# The tangents are permuted like the values: sorted along with the key.
prim_sort[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  with_tangent <- which(!vapply(tangents, is.null, logical(1L)))
  sorted <- prim_sort(
    c(inputs[1L], tangents[with_tangent]),
    axis = params$axis,
    decreasing = params$decreasing,
    stable = params$stable
  )
  out <- vector("list", length(outputs))
  out[with_tangent] <- sorted[-1L]
  out
}

# The tangent of the values is that of the elements of `x` they are, at their
# indices, where the pullback rule scatters the cotangent to.
prim_top_k[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  indices <- if (params$indices) outputs[[2L]] else prim_top_k(inputs[[1L]], k = params$k, indices = TRUE)[[2L]]
  out <- list(take_along_axis(tangents[[1L]], indices, axis = 1L))
  if (params$indices) {
    out[2L] <- list(NULL)
  }
  out
}

# Pushforward rule of scatter --------------------------------------------------

# Differentiates the update functions the pullback rule does, under the same
# conventions: the update that is written, or `x` and the updates that attain
# an extremum, give the tangent of a position.
prim_scatter[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  x <- inputs[[1L]]
  update <- inputs[[3L]]
  has_x <- !is.null(tangents[[1L]])
  has_update <- !is.null(tangents[[3L]])
  kind <- scatter_derivative_kind(params, has_update, "pushforward")
  if (!has_x && !has_update) {
    return(list(NULL))
  }
  at <- scatter_positions(params, inputs[[2L]], shape(update), shape(x))
  t_x <- tangents[[1L]] %||% zeros_like(x)
  t_update <- tangents[[3L]] %||% zeros_like(update)

  tangent <- switch(
    kind,
    replace = if (params$unique_indices) {
      at$scatter_at(t_x, t_update, "replace")
    } else {
      # `x` keeps its tangent where no update is written, and the update that
      # is written gives its own.
      winners <- scatter_replace_winners(at, shape(update), params$update_window_axes, shape(x))
      at$scatter_at(at$scatter_at(t_x, zeros_like(update), "replace"), prim_ifelse(winners, t_update, 0), "add", FALSE)
    },
    add = ,
    sub = at$scatter_at(t_x, t_update, kind),
    mul = {
      # result = x * (product of the updates to the position), and with unique
      # indices x * update at the update's position
      terms <- list(
        if (has_x) at$scatter_at(t_x, update, "mul"),
        if (has_update) prim_mul(x, at$scatter_at(zeros_like(x), t_update, "add"))
      )
      Reduce(prim_add, Filter(Negate(is.null), terms))
    },
    max = ,
    min = {
      w <- scatter_extremum_winners(x, update, outputs[[1L]], at, dtype(t_x))
      won <- at$scatter_at(zeros_like(x), prim_mul(w$won, t_update), "add")
      prim_div(prim_add(prim_mul(t_x, w$retained), won), w$count)
    }
  )
  list(tangent)
}

# Pushforward rules of the linear algebra primitives ---------------------------
#
# These take batches of matrices: the first two axes form the matrix, any
# trailing ones are batch axes, so `t()` is `transpose_matrix_axes()`. The
# helpers they share with the pullback rules are in `rules-pullback.R`.

# The factor of a symmetric `A = L t(L)` moves along a symmetric `dA` by
# `dL = L Φ(L^-1 dA t(L)^-1)` (Murray 2016). Like the pullback rule, this takes
# the factor as a function of the symmetric matrices, so a tangent counts by
# its symmetric part.
prim_chol[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  L <- outputs[[1L]]
  if (!params$lower) {
    L <- transpose_matrix_axes(L)
  }
  dA <- symmetric_part(tangents[[1L]])
  dL <- nv_matmul(L, cholesky_phi(solve_both_sides(L, dA, transpose = FALSE)))
  list(if (params$lower) dL else transpose_matrix_axes(dL))
}

# `op(a) x = b` moves by `op(a) dx = db - op(da) x`, and `x op(a) = b` by
# `dx op(a) = db - x op(da)`: one more solve against `a`. Only the entries of
# `a` the solve reads count, so only those of `da` do, as in the pullback rule.
prim_triangular_solve[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  da <- tangents[[1L]]
  rhs <- tangents[[2L]]
  if (!is.null(da)) {
    da <- prim_ifelse(triangular_mask(shape(da), params$lower, params$unit_diagonal), da, 0)
    if (params$transpose_a) {
      da <- transpose_matrix_axes(da)
    }
    da_x <- matmul_on_side(da, outputs[[1L]], params$left_side)
    rhs <- if (is.null(rhs)) prim_negate(da_x) else prim_sub(rhs, da_x)
  }
  list(do.call(prim_triangular_solve, c(list(inputs[[1L]], rhs), params)))
}

# Pushforward rules of the higher-order primitives -----------------------------
#
# As JAX's `jvp` of `cond`, `scan` and `while_loop`: the tangents of a call
# that runs sub-graphs come from the same primitive, called anew with
# sub-graphs that push the tangents of their inputs through alongside the
# values (`push_through()`). A loop carries the tangents of the values it
# carries as more state, but only of those a tangent reaches
# (`carry_closure()`), so that a counter or a random state carries none. The
# new call computes the values its tangents need itself, as the replayed
# primal call keeps none of those of its steps or branches.

# Which of `tangents` are not `NULL`.
has_tangent <- function(tangents) {
  !vapply(tangents, is.null, logical(1L))
}

# The tangents of the branch the predicate selects: an `if` on the same
# predicate whose branches push the tangents of the operands through those of
# the call. An output that a tangent reaches in neither branch gets none.
prim_if[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  operands <- inputs[-1L]
  operand_tangents <- tangents[-1L]
  branches <- list(params$true, params$false)
  want <- Reduce(`|`, lapply(branches, graph_requiring_outputs, seeds = has_tangent(operand_tangents)))
  out <- vector("list", length(outputs))
  if (any(want)) {
    branch <- function(graph) function() push_through(graph, operands, operand_tangents, want)$tangent
    out[want] <- prim_if(inputs[[1L]], branch(params$true), branch(params$false))
  }
  out
}

# A scan over the carry and the tangents it carries, and over `xs` and their
# tangents, whose step pushes the tangents through the body of the call. A
# carried value that enters without a tangent, but that the body gives one,
# starts from a zero. The scan stacks the tangents of the outputs; their values
# are the call's.
prim_scan[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  body <- params$body
  carry_idx <- seq_len(params$n_carry)
  xs_idx <- params$n_carry + seq_len(params$n_xs)
  with_tangent <- has_tangent(tangents)
  carry_t <- carry_closure(body, with_tangent[carry_idx], with_tangent[-carry_idx], graph_requiring_outputs)
  out_t <- graph_requiring_outputs(body, c(carry_t, with_tangent[-carry_idx]))[-carry_idx]
  out <- vector("list", length(outputs))
  if (!any(carry_t, out_t)) {
    return(out)
  }
  xs_t <- xs_idx[with_tangent[xs_idx]]
  n_carry_t <- sum(carry_t)
  scanned <- prim_scan(
    init = list(
      value = inputs[carry_idx],
      tangent = Map(\(x, t) t %||% zeros_like(x), inputs[carry_idx][carry_t], tangents[carry_idx][carry_t])
    ),
    xs = list(value = inputs[xs_idx], tangent = tangents[xs_t]),
    body = function(carry, x) {
      # the captures and their tangents as they are, the carry and `xs` of
      # the step
      step_tangents <- tangents
      step_tangents[c(carry_idx, xs_idx)] <- list(NULL)
      step_tangents[carry_idx[carry_t]] <- carry$tangent
      step_tangents[xs_t] <- x$tangent
      operands <- c(carry$value, x$value, inputs[-c(carry_idx, xs_idx)])
      pushed <- push_through(body, operands, step_tangents, c(carry_t, out_t))
      list(
        carry = list(value = pushed$value[carry_idx], tangent = pushed$tangent[seq_len(n_carry_t)]),
        out = pushed$tangent[n_carry_t + seq_len(sum(out_t))]
      )
    },
    steps = params$steps,
    reverse = params$reverse
  )
  out[carry_idx[carry_t]] <- scanned$carry$tangent
  out[params$n_carry + which(out_t)] <- scanned$out
  out
}

# A while loop over the state and the tangents it carries, whose condition
# reads the state as the call's does, and whose body pushes the tangents
# through the body of the call. A value of the state that enters without a
# tangent, but that the body gives one, starts from a zero. Unlike reverse
# mode, this needs nothing of the iterations but the state of the current one,
# so the number of iterations need not be known.
prim_while[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  state_idx <- seq_len(length(inputs) - params$n_captures)
  captures <- inputs[-state_idx]
  with_tangent <- has_tangent(tangents)
  state_t <- carry_closure(params$body, with_tangent[state_idx], with_tangent[-state_idx], graph_requiring_outputs)
  out <- vector("list", length(outputs))
  if (!any(state_t)) {
    return(out)
  }
  t_idx <- which(state_t)
  tangent_idx <- length(state_idx) + seq_along(t_idx)
  init <- c(inputs[state_idx], Map(\(x, t) t %||% zeros_like(x), inputs[t_idx], tangents[t_idx]))
  names(init) <- c(paste0("s", state_idx), paste0("t", t_idx))
  cond <- graph_without_prints(params$cond)
  looped <- prim_while(
    init,
    cond = function(...) graph_apply(cond, c(unname(list(...))[state_idx], captures))[[1L]],
    body = function(...) {
      state <- unname(list(...))
      step_tangents <- tangents
      step_tangents[state_idx] <- list(NULL)
      step_tangents[t_idx] <- state[tangent_idx]
      pushed <- push_through(params$body, c(state[state_idx], captures), step_tangents, state_t)
      stats::setNames(c(pushed$value, pushed$tangent), names(init))
    }
  )
  out[t_idx] <- unname(looped)[tangent_idx]
  out
}

# The custom gradient only replaces reverse mode: the tangents are those of
# `fn`, pushed through its calls.
prim_custom_gradient[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  want <- graph_requiring_outputs(params$fn, has_tangent(tangents))
  out <- vector("list", length(outputs))
  if (any(want)) {
    out[want] <- push_through(params$fn, inputs, tangents, want)$tangent
  }
  out
}
