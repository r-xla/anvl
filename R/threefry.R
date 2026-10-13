# Random bits with ThreeFry-2x32 (Salmon et al., 2011), built from StableHLO
# ops rather than XLA's `rng_bit_generator`: its output does not depend on how a
# backend or compiler version lowers that op, and the advanced state is computed
# from the counter alone, so a later draw does not depend on the bits of an
# earlier one.
#
# A key is a `ui64`, whose low and high 32 bits are the two words of the
# ThreeFry key. A key is used either to draw from or to split, never both: a
# split hashes the key with the blocks a draw from it would hash with counters
# that are multiples of 2^32.

# A draw of the unsigned `dtype` and `shape` from the RNG state `state`, a
# `ui64[2]` FuncValue holding the key and the counter, laid out as XLA's
# THREE_FRY `rng_bit_generator` lays out its state: a `ui64` draw of a 1-D shape
# is what that op draws. Each value hashes a counter value of its own, whatever
# the dtype. Returns `list(values, state)`, the draw and the state with its
# counter advanced by the number of values drawn. The values fill `shape` in
# column-major order.
hlo_threefry_bits <- function(state, shape, dtype) {
  func <- state$func
  shape <- as.integer(shape)
  dtype <- as_dtype(dtype)

  element <- function(i) hlo_reshape(hlo_slice(state, i, i + 1L, 1L), integer())
  key <- element(0L)
  counter <- element(1L)

  counters <- hlo_add(hlo_broadcast_in_dim(counter, integer(), shape), threefry_index(shape, func))
  keys <- hlo_broadcast_in_dim(key, integer(), shape)
  bits <- threefry2x32(threefry_words(keys, shape, func), threefry_words(counters, shape, func), shape, func)
  values <- switch(
    as.character(dtype),
    ui64 = threefry_join(bits, shape, func),
    ui32 = hlo_xor(bits[[1L]], bits[[2L]]),
    {
      # the low bits, masked so that the conversion never overflows, whose
      # result StableHLO leaves undefined
      mask <- c(ui8 = 0xFFL, ui16 = 0xFFFFL)[[as.character(dtype)]]
      hlo_convert(hlo_and(hlo_xor(bits[[1L]], bits[[2L]]), threefry_fill(mask, "ui32", shape, func)), dtype)
    }
  )

  advanced <- hlo_add(counter, threefry_u64(prod(shape), func))
  new_state <- hlo_concatenate(
    hlo_reshape(key, 1L),
    hlo_reshape(advanced, 1L),
    dimension = 0L
  )
  list(values, new_state)
}

# The keys of shape `shape` that splitting the `ui64` scalar key `key` gives,
# as JAX's `jax.random.split()` derives them (with `jax_threefry_partitionable`,
# its default): the key with index `i` is the block that hashing `key` with the
# block (high, low 32 bits of `i`) gives. The index is the column-major linear
# index into `shape`, so a 1-D split gives the keys JAX's does.
hlo_threefry_split <- function(key, shape) {
  func <- key$func
  shape <- as.integer(shape)
  index <- threefry_words(threefry_index(shape, func), shape, func)
  keys <- hlo_broadcast_in_dim(key, integer(), shape)
  bits <- threefry2x32(threefry_words(keys, shape, func), rev(index), shape, func)
  threefry_join(bits, shape, func)
}

# ThreeFry-2x32 with 20 rounds (Salmon et al., 2011), as XLA and JAX implement
# it: the key `key` and the input block `x` are each two `ui32` arrays (its
# first and second word) of shape `shape`, and so is the result.
threefry2x32 <- function(key, x, shape, func) {
  const <- function(value) threefry_fill(value, "ui32", shape, func)
  rotate_left <- function(v, d) {
    hlo_or(hlo_shift_left(v, const(d)), hlo_shift_right_logical(v, const(32L - d)))
  }
  rotations <- list(c(13L, 15L, 26L, 6L), c(17L, 29L, 16L, 24L))
  rounds <- function(x, r) {
    for (d in rotations[[r]]) {
      x[[1L]] <- hlo_add(x[[1L]], x[[2L]])
      x[[2L]] <- hlo_xor(x[[1L]], rotate_left(x[[2L]], d))
    }
    x
  }

  # 0x1BD11BDA is the parity constant of the key schedule
  ks <- list(key[[1L]], key[[2L]], hlo_xor(hlo_xor(const(0x1BD11BDAL), key[[1L]]), key[[2L]]))
  x[[1L]] <- hlo_add(x[[1L]], ks[[1L]])
  x[[2L]] <- hlo_add(x[[2L]], ks[[2L]])
  for (i in seq_len(5L)) {
    x <- rounds(x, (i - 1L) %% 2L + 1L)
    x[[1L]] <- hlo_add(x[[1L]], ks[[i %% 3L + 1L]])
    x[[2L]] <- hlo_add(hlo_add(x[[2L]], ks[[(i + 1L) %% 3L + 1L]]), const(i))
  }
  x
}

# The column-major linear index into `shape`, as a `ui64` array of that shape.
threefry_index <- function(shape, func) {
  out <- threefry_fill(0L, "ui64", shape, func)
  stride <- 1
  for (axis in seq_along(shape)) {
    iota <- hlo_iota(iota_dimension = axis - 1L, dtype = "ui64", shape = shape)
    if (stride != 1) {
      iota <- hlo_multiply(iota, hlo_broadcast_in_dim(threefry_u64(stride, func), integer(), shape))
    }
    out <- hlo_add(out, iota)
    stride <- stride * shape[[axis]]
  }
  out
}

# The `ui64` array `x` of shape `shape` as its low and high 32 bits.
threefry_words <- function(x, shape, func) {
  hi <- hlo_shift_right_logical(x, threefry_fill(32L, "ui64", shape, func))
  list(hlo_convert(x, "ui32"), hlo_convert(hi, "ui32"))
}

# The low and high 32 bits `x`, two `ui32` arrays of shape `shape`, as one
# `ui64` array.
threefry_join <- function(x, shape, func) {
  hi <- hlo_shift_left(hlo_convert(x[[2L]], "ui64"), threefry_fill(32L, "ui64", shape, func))
  hlo_or(hlo_convert(x[[1L]], "ui64"), hi)
}

# `value` as an array of the unsigned `dtype` and `shape`.
threefry_fill <- function(value, dtype, shape, func) {
  hlo_broadcast_in_dim(hlo_scalar(value, dtype = dtype, func = func), integer(), shape)
}

# The whole number `value` (below 2^53) as a `ui64` scalar. An R double does not
# reach `hlo_scalar()` intact past the integer range, so the value is built from
# two 30-bit halves.
threefry_u64 <- function(value, func) {
  lo <- hlo_scalar(as.integer(value %% 2^30), dtype = "ui64", func = func)
  if (value < 2^30) {
    return(lo)
  }
  hi <- hlo_scalar(as.integer(value %/% 2^30), dtype = "ui64", func = func)
  hlo_or(hlo_shift_left(hi, hlo_scalar(30L, dtype = "ui64", func = func)), lo)
}
