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

# `prim_sum()` is linear: the tangent of the result is the sum of the operand's
# tangent over the same axes.
prim_sum[["pushforward"]] <- function(inputs, outputs, tangents, params) {
  list(prim_sum(tangents[[1L]], axes = params$axes, drop = params$drop))
}
