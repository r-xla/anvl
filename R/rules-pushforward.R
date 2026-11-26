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
    da <- keep_triangle(da, params$lower, params$unit_diagonal)
    if (params$transpose_a) {
      da <- transpose_matrix_axes(da)
    }
    da_x <- matmul_on_side(da, outputs[[1L]], params$left_side)
    rhs <- if (is.null(rhs)) prim_negate(da_x) else prim_sub(rhs, da_x)
  }
  list(do.call(prim_triangular_solve, c(list(inputs[[1L]], rhs), params)))
}
