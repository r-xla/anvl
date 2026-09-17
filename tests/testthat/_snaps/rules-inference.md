# assert_array_dtype() / names the categories it wanted and the data type it got

    Code
      assert_array_dtype(infer_at("i32", 3L), "float", arg = "x")
    Condition
      Error in `assert_array_dtype()`:
      ! `x` must have dtype float.
      x Got "i32".

# assert_array_dtype() / checks shape and rank when asked

    Code
      assert_array_dtype(infer_at("f32", c(2L, 3L)), shape = integer(), arg = "pred")
    Condition
      Error in `assert_array_dtype()`:
      ! `pred` must have shape ().
      x Got (2x3).

---

    Code
      assert_array_dtype(infer_at("f32", c(2L, 3L)), naxes = 1L, arg = "initial_state")
    Condition
      Error in `assert_array_dtype()`:
      ! `initial_state` must have 1 axis.
      x Got 2.

# the element-wise rules / refuse operands whose types disagree

    Code
      infer_generic_biv(infer_at("i32", 4L), infer_at("i32", 6L))
    Condition
      Error in `assert_same_type()`:
      ! `lhs` and `rhs` must have the same array type.
      x Got i32[4] and i32[6].

# infer_transpose() / reports the permutation it expected in 1-based axes

    Code
      infer_transpose(infer_at("f32", c(2L, 2L)), 1L)
    Condition
      Error in `infer_transpose()`:
      ! `permutation` must be a permutation of c(1, 2).
      x Got 1.

# infer_broadcast_in_axes() / refuses an axis that is neither 1 nor the target size

    Code
      infer_broadcast_in_axes(infer_at("f32", c(2L, 3L)), c(4L, 3L), c(1L, 2L))
    Condition
      Error in `infer_broadcast_in_axes()`:
      ! Axis 1 of `x` must be 4 or 1 to broadcast to axis 1 of the result.
      x Got shapes (2x3) and (4x3).

# infer_static_slice() / refuses a stride of zero rather than computing an infinite shape

    Code
      infer_static_slice(infer_at("i32", 10L), 1L, 5L, 0L)
    Condition
      Error in `infer_static_slice()`:
      ! `strides` must be positive.
      x Got 0.

# infer_static_slice() / refuses a limit past the end of the array

    Code
      infer_static_slice(infer_at("i32", 10L), 1L, 11L, 1L)
    Condition
      Error in `infer_static_slice()`:
      ! `limit_indices` must not exceed the shape of `x` ((10)).
      x Got 11 at axis 1.

# infer_concatenate() / refuses inputs that disagree on any other axis

    Code
      infer_concatenate(infer_at("f32", c(2L, 3L)), infer_at("f32", c(2L, 4L)), axis = 1L)
    Condition
      Error in `infer_concatenate()`:
      ! Every input must have the same shape except along `axis` (1).
      x Got (2x3) and (2x4).

# infer_dot_general() / refuses contracted axes whose sizes differ

    Code
      infer_dot_general(infer_at("f32", c(2L, 3L)), infer_at("f32", c(4L, 5L)),
      contracting_axes = list(2L, 1L), batching_axes = list(integer(), integer()),
      precision = "highest")
    Condition
      Error in `infer_dot_general()`:
      ! The contracted axes of `lhs` and `rhs` must have the same sizes.
      x Axes 2 of (2x3) are 3, axes 1 of (4x5) are 4.

# infer_pad() / refuses negative padding that would empty an axis

    Code
      infer_pad(infer_at("f32", 3L), infer_at("f32"), -3L, -3L, 0L)
    Condition
      Error in `infer_pad()`:
      ! Negative padding must not remove more than an axis holds.
      x Axis 1 of `x` ((3)) would end up at -3.

# infer_top_k() / refuses a k larger than the last axis

    Code
      infer_top_k(infer_at("f32", c(2L, 3L)), k = 4L)
    Condition
      Error in `infer_top_k()`:
      ! `k` must not exceed the size of the last axis of `x` (3).
      x Got 4.

# the inference rules as the primitives reach them / reports an axis the primitive itself does not catch in 1-based terms

    Code
      jit(prim_transpose, static = "permutation")(nv_array(1:4, shape = c(2, 2)),
      permutation = 1L)
    Condition
      Error in `prim_transpose()`:
      ! `permutation` must be a permutation of c(1, 2).
      x Got 1.

