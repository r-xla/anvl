# assert_array_dtype() / names the categories it wanted and the data type it got

    Code
      assert_array_dtype(infer_at("i32", 3L), "float", arg = "x")
    Condition
      Error in `assert_array_dtype()`:
      ! `x` must have a float data type.
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
      ! `limit_indices` must not exceed the shape of `x` (10).
      x Got 11 at axis 1.

# infer_concatenate() / refuses inputs that disagree on any other axis

    Code
      infer_concatenate(infer_at("f32", c(2L, 3L)), infer_at("f32", c(2L, 4L)), axis = 1L)
    Condition
      Error in `infer_concatenate()`:
      ! Every input must have the same shape except along `axis` (1).
      x Got (2x3) and (2x4).

# infer_concatenate() / refuses inputs with a different number of axes

    Code
      infer_concatenate(infer_at("f32", c(2L, 3L, 4L)), infer_at("f32", c(2L, 3L)),
      axis = 3L)
    Condition
      Error in `infer_concatenate()`:
      ! Every input must have the same number of axes.
      x Input 1 has 3 axes (2x3x4), input 2 has 2 (2x3).

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
      x `x` is (3); axis 1 would end up at -3.

# infer_top_k() / refuses a k larger than the last axis

    Code
      infer_top_k(infer_at("f32", c(2L, 3L)), k = 4L, indices = TRUE)
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

# infer_convolution() / refuses a padding that takes away more than an axis holds

    Code
      nv_conv2d(x, k, padding = -4L)
    Condition
      Error in `prim_convolution()`:
      ! `padding` must not remove more than axis 3 of `x` holds.
      x Axis 3 dilates to 5, and padding -4 and -4 leaves -3.

# infer_convolution() / names `precision` rather than leaving it to the lowering

    Code
      nv_conv2d(x, k, precision = "nope")
    Condition
      Error in `prim_convolution()`:
      ! `precision` must be one of "default", "high", or "highest".
      x Got "nope".

# infer_convolution() / refuses a kernel with a size-0 spatial axis

    Code
      jit(prim_convolution, static = 3:18)(nv_array(as.double(1:50), shape = c(1, 2,
        5, 5)), nv_array(numeric(), shape = c(3, 2, 0, 3)), 1L, 2L, c(3L, 4L), 2L, 1L,
      c(3L, 4L), 1L, 2L, c(3L, 4L), c(1L, 1L), matrix(0L, 2L, 2L), c(1L, 1L), c(1L,
        1L), 1L, 1L, "highest")
    Condition
      Error in `prim_convolution()`:
      ! `kernel` must not have a size-0 spatial axis.
      x Axis 3 of `kernel` has size 0.

# infer_convolution() / blames the layout, not `padding`, when the rank disagrees

    Code
      jit(prim_convolution, static = 3:18)(nv_array(as.double(1:10), shape = c(1, 2,
        5)), nv_array(as.double(1:18), shape = c(3, 2, 3)), 1L, 2L, c(3L, 4L), 2L, 1L,
      c(3L, 4L), 1L, 2L, c(3L, 4L), c(1L, 1L), matrix(0L, 2L, 2L), c(1L, 1L), c(1L,
        1L), 1L, 1L, "highest")
    Condition
      Error in `prim_convolution()`:
      ! `input_spatial_axes` must have one entry per spatial axis (1).
      x Got 2.

# infer_while() / checks what `cond` returns, not only the body

    Code
      prim_while(init = list(i = nv_scalar(0L)), cond = function(i) nv_convert(i,
        "f32"), body = function(i) list(i = i + 1L))
    Condition
      Error in `prim_while()`:
      ! `cond` must return a boolean scalar.
      x Got f32[].

---

    Code
      prim_while(init = list(i = nv_array(c(1L, 2L))), cond = function(i) i < 5L,
      body = function(i) list(i = i + 1L))
    Condition
      Error in `prim_while()`:
      ! `cond` must return a boolean scalar.
      x Got bool[2].

# infer_cum() / refuses a scan over a size-0 axis

    Code
      prim_cumsum(nv_fill(1, shape = c(0, 3)), axis = 1L)
    Condition
      Error in `prim_cumsum()`:
      ! `x` must have elements along the axis it accumulates over.
      x `x` has shape (0x3); axis 1 has size 0.

# infer_rng_bit_generator() / fixes the state length DEFAULT needs

    Code
      prim_rng_bit_generator(nv_array(rep(0, 5), dtype = "ui64"), rng_algorithm = "DEFAULT",
      dtype = "f32", shape = c(3, 2))
    Condition
      Error in `prim_rng_bit_generator()`:
      ! "DEFAULT" requires an `initial_state` of length 3.
      x Got 5.
      i Name "THREE_FRY" or "PHILOX" to use a shorter state.

# infer_gather() / names each argument of an overlapping axis pair

    Code
      prim_gather(nv_matrix(1:9, nrow = 3), nv_array(rep(1L, 4), shape = c(2L, 2L)),
      slice_sizes = c(1L, 1L), offset_axes = integer(), collapsed_slice_axes = c(1L,
        2L), x_batching_axes = integer(), start_indices_batching_axes = integer(),
      start_index_map = c(1L, 1L), index_vector_axis = 2L)
    Condition
      Error in `prim_gather()`:
      ! `start_index_map` must contain unique axes.
      x Axis 1 is named 2 times.

# infer_gather() / reports an axis of 0 against the argument that holds it

    Code
      prim_gather(nv_matrix(1:9, nrow = 3), nv_matrix(c(1L, 3L), ncol = 1),
      slice_sizes = c(1L, 3L), offset_axes = 2L, collapsed_slice_axes = 0L,
      x_batching_axes = integer(), start_indices_batching_axes = integer(),
      start_index_map = 1L, index_vector_axis = 2L)
    Condition
      Error in `prim_gather()`:
      ! `collapsed_slice_axes` must contain axes between 1 and 2.
      x Got 0.

# assert_subgraph_closed() / refuses a reductor that reads a value from around it

    Code
      jit(function(a, y) {
        prim_reduce(a, init = 0, axes = 1L, reductor = function(p, q) p + q + y)
      })(x, nv_scalar(1))
    Condition
      Error in `prim_reduce()`:
      ! `reductor` must use only the values it is given.
      x It reads f32[] from the function around it.
      i The region it becomes takes a fixed set of operands, so there is nowhere to pass that value in.

# assert_subgraph_closed() / refuses an update computation that reads a value from around it

    Code
      jit(function(a, b, c, y) {
        prim_scatter(a, b, c, update_window_axes = integer(), inserted_window_axes = 1L,
        x_batching_axes = integer(), scatter_indices_batching_axes = integer(),
        scatter_axes_to_x_axes = 1L, index_vector_axis = 2L, update_computation = function(
          old, new) new + y)
      })(nv_array(c(0, 0, 0, 0, 0)), nv_matrix(c(1L, 3L), ncol = 1), nv_array(c(10,
        30)), nv_scalar(5))
    Condition
      Error in `prim_scatter()`:
      ! `update_computation` must use only the values it is given.
      x It reads f32[] from the function around it.
      i The region it becomes takes a fixed set of operands, so there is nowhere to pass that value in.

# the reduce rules / refuse a reductor that does not return a scalar

    Code
      prim_reduce(x, init = 0, axes = 1L, reductor = function(a, b) nv_fill(0, shape = c(
        2, 2)))
    Condition
      Error in `prim_reduce()`:
      ! `reductor` must return a scalar.
      x Got shape (2x2).

