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

# Pushforward rules of the linear primitives ----------------------------------
#
# A primitive tagged `"linear"` is linear in its float operands, its other
# operands held fixed, so the tangent of its result is the primitive applied to
# the tangents of its float operands -- zeros for one without a tangent -- with
# its other operands and its parameters as they are.

# The pushforward rule of the linear primitive `prim`.
pushforward_linear <- function(prim) {
  prim_fn <- primitive_def(prim)$fn
  function(inputs, outputs, tangents, params) {
    operands <- Map(
      function(x, tangent) {
        if (is_dtype_float(dtype(x))) tangent %||% zeros(dtype(x), shape(x)) else x
      },
      inputs,
      tangents
    )
    if (!all(names(params) %in% formalArgs(prim_fn))) {
      cli_abort(
        "Internal error: the linear primitive {.fn {paste0('prim_', primitive_def(prim)$name)}} cannot be called with its parameters."
      )
    }
    out <- do.call(prim_fn, c(operands, params))
    if (is_graph_box(out)) list(out) else out
  }
}

for (prim in primitives_tagged("linear")) {
  prim[["pushforward"]] <- pushforward_linear(prim)
}
rm(prim)

# The barrier passes its operands through, and takes their tangents through a
# barrier of their own, as its pullback rule does their cotangents.
prim_optimization_barrier[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  has_tangent <- !vapply(tangents, is.null, logical(1L))
  out <- vector("list", length(inputs))
  out[has_tangent] <- prim_optimization_barrier(tangents[has_tangent])
  out
}
