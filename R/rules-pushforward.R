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
