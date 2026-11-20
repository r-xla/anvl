#' @include rules-pullback.R
NULL

# Pushforward rules of the elementwise primitives ------------------------------
#
# An element of the result of an elementwise primitive depends only on the
# elements at the same position of its operands, so its derivative with respect
# to each operand is diagonal: its own transpose. The pullback rule multiplies a
# cotangent of the result by that derivative, so given a tangent of an operand
# in place of the cotangent, it gives that tangent's pushforward. The
# pushforward rule of a primitive created with `new_primitive(elementwise =
# TRUE)` therefore calls its pullback rule once per operand with a tangent, and
# adds up the results.

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

local({
  for (name in ls(primitive_env)) {
    def <- primitive_def(primitive_env[[name]])
    if (isTRUE(def$elementwise)) {
      if (is.null(def[["pullback"]]$backward)) {
        cli_abort(
          "Internal error: the elementwise primitive {.field {name}} has no pullback rule without a forward pass."
        )
      }
      def[["pushforward"]] <- pushforward_from_pullback(def[["pullback"]])
    }
  }
})
