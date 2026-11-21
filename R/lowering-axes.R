# Axis reversal in the StableHLO lowering.
#
# anvl is column-major: an array of shape (d1, ..., dn) keeps its elements with
# the first axis varying fastest. XLA's arrays are row-major. The two agree once
# every axis is reversed: R's column-major bytes of (d1, ..., dn) are exactly
# XLA's row-major bytes of (dn, ..., d1). So the lowering represents an array
# of R shape (d1, ..., dn) as an XLA array of shape (dn, ..., d1), and the pjrt
# backend stores it that way, which makes every upload and read-back a plain
# copy. R axis `a` (1-based) of an `n`-axis array is XLA dimension `n - a`
# (0-based), and R axis 1, the contiguous one, is XLA's minor dimension.
#
# `at2vt()` reverses the shapes of every type it builds, so the function
# signature, the regions and the output types a rule receives are all in XLA
# order, and so is `shape()` of every `FuncValue` a rule sees. The rules map
# the axis numbers of their parameters with the helpers below.

# The XLA dimensions (0-based) of the R axes `axes` (1-based) of an `n`-axis
# array.
xla_dims <- function(axes, n) {
  as.integer(n - axes)
}

# The XLA shape of the R shape `shape`.
xla_shape <- function(shape) {
  rev(as.integer(shape))
}

# The R shape of a `FuncValue` (whose type holds the XLA shape).
r_shape <- function(x) {
  rev(shape(x$value_type))
}

# The XLA permutation of the R permutation `perm`, where axis i of the result
# is axis `perm[i]` of the operand.
xla_perm <- function(perm) {
  n <- length(perm)
  as.integer(n - rev(perm))
}

# The `broadcast_dimensions` of an XLA broadcast of an operand into an `n`-axis
# result, where R axis j of the operand becomes R axis `axes[j]` of the result.
# XLA indexes them by the operand's XLA dimension, so their order reverses too.
xla_broadcast_dims <- function(axes, n) {
  xla_dims(rev(axes), n)
}

# A per-axis vector (one entry per R axis) in XLA dimension order.
xla_per_axis <- function(x) {
  rev(x)
}

# The batching dimensions of a gather or scatter: R axes `x_axes` of the
# `n_x`-axis operand paired with R axes `indices_axes` of the `n_indices`-axis
# indices, as XLA dimensions sorted by the operand's (stablehlo requires them
# sorted) with each pair kept together.
xla_batching_pairs <- function(x_axes, n_x, indices_axes, n_indices) {
  x_dims <- xla_dims(x_axes, n_x)
  indices_dims <- xla_dims(indices_axes, n_indices)
  o <- order(x_dims)
  list(x_dims[o], indices_dims[o])
}

# The XLA `index_vector_dim` of a gather or scatter whose `n_indices`-axis
# indices hold the index vectors along R axis `axis`. Axis `n_indices + 1` is an
# implicit trailing axis of size 1; an axis of size 1 can sit anywhere without
# reordering the elements, so it becomes XLA's implicit trailing dimension.
xla_index_vector_dim <- function(axis, n_indices) {
  if (axis > n_indices) n_indices else xla_dims(axis, n_indices)
}
