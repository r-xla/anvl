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
