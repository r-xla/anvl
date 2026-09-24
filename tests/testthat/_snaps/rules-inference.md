# assert_array_dtype() / names the categories it wanted and the data type it got

    Code
      assert_array_dtype(nv_aval("i32", 3L), "float", arg = "x")
    Condition
      Error in `assert_array_dtype()`:
      ! `x` must have a float data type.
      x Got "i32".

# assert_array_dtype() / checks shape and rank when asked

    Code
      assert_array_dtype(nv_aval("f32", c(2L, 3L)), shape = integer(), arg = "pred")
    Condition
      Error in `assert_array_dtype()`:
      ! `pred` must have shape ().
      x Got (2x3).

---

    Code
      assert_array_dtype(nv_aval("f32", c(2L, 3L)), naxes = 1L, arg = "initial_state")
    Condition
      Error in `assert_array_dtype()`:
      ! `initial_state` must have 1 axis.
      x Got (2x3).

# the element-wise rules / refuse operands whose types disagree

    Code
      infer_generic_biv(nv_aval("i32", 4L), nv_aval("i32", 6L))
    Condition
      Error in `assert_same_type()`:
      ! `lhs` and `rhs` must have the same array type.
      x Got i32[4] and i32[6].

# infer_transpose() / reports the permutation it expected in 1-based axes

    Code
      infer_transpose(nv_aval("f32", c(2L, 2L)), 1L)
    Condition
      Error in `infer_transpose()`:
      ! `perm` must be a permutation of c(1, 2).
      x Got 1.

# infer_broadcast_in_axes() / refuses an axis that is neither 1 nor the target size

    Code
      infer_broadcast_in_axes(nv_aval("f32", c(2L, 3L)), c(4L, 3L), c(1L, 2L))
    Condition
      Error in `infer_broadcast_in_axes()`:
      ! Axis 1 of `x` must be 4 or 1 to broadcast to axis 1 of the result.
      x Got shapes (2x3) and (4x3).

# infer_static_slice() / refuses a stride of zero rather than computing an infinite shape

    Code
      infer_static_slice(nv_aval("i32", 10L), 1L, 5L, 0L)
    Condition
      Error in `infer_static_slice()`:
      ! `strides` must be positive.
      x Got 0.

# infer_static_slice() / refuses an end index past the end of the array

    Code
      infer_static_slice(nv_aval("i32", 10L), 1L, 11L, 1L)
    Condition
      Error in `infer_static_slice()`:
      ! `end_indices` must not exceed the shape of `x` (10).
      x Got 11 at axis 1.

# infer_concatenate() / refuses inputs that disagree on any other axis

    Code
      infer_concatenate(nv_aval("f32", c(2L, 3L)), nv_aval("f32", c(2L, 4L)), axis = 1L)
    Condition
      Error in `infer_concatenate()`:
      ! Every input must have the same shape except along `axis` (1).
      x `..1` has shape (2x3), `..2` has shape (2x4).

# infer_concatenate() / refuses inputs with a different number of axes

    Code
      infer_concatenate(nv_aval("f32", c(2L, 3L, 4L)), nv_aval("f32", c(2L, 3L)),
      axis = 3L)
    Condition
      Error in `infer_concatenate()`:
      ! Every input must have the same number of axes.
      x `..1` has 3 axes (2x3x4), `..2` has 2 (2x3).

# infer_dot_general() / refuses contracted axes whose sizes differ

    Code
      infer_dot_general(nv_aval("f32", c(2L, 3L)), nv_aval("f32", c(4L, 5L)),
      contracting_axes = list(2L, 1L), batching_axes = list(integer(), integer()),
      precision = "highest")
    Condition
      Error in `infer_dot_general()`:
      ! The contracted axes of `lhs` and `rhs` must have the same sizes.
      x Axes 2 of (2x3) are 3, axes 1 of (4x5) are 4.

# infer_pad() / refuses negative padding that would empty an axis

    Code
      infer_pad(nv_aval("f32", 3L), nv_aval("f32", integer()), -3L, -3L, 0L)
    Condition
      Error in `infer_pad()`:
      ! Negative padding must not remove more than an axis holds.
      x `x` has shape (3); axis 1 would end up at -3.
      i Got `edge_padding_low` = -3, `edge_padding_high` = -3, `interior_padding` = 0.

# infer_top_k() / refuses a k larger than the last axis

    Code
      infer_top_k(nv_aval("f32", c(2L, 3L)), k = 4L, indices = TRUE)
    Condition
      Error in `infer_top_k()`:
      ! `k` must not exceed the size of the last axis of `x` (3).
      x Got 4.

# the inference rules as the primitives reach them / reports an axis the primitive itself does not catch in 1-based terms

    Code
      jit(prim_transpose, static = "perm")(nv_array(1:4, shape = c(2, 2)), perm = 1L)
    Condition
      Error in `prim_transpose()`:
      ! `perm` must be a permutation of c(1, 2).
      x Got 1.

# prim_reshape

    Code
      prim_reshape(nv_array(1:4), shape = "a")
    Condition
      Error in `prim_reshape()`:
      ! `shape` must be a whole number vector.
      x Got "a".

---

    Code
      prim_reshape(nv_array(1:4), shape = c(3L, 3L))
    Condition
      Error in `prim_reshape()`:
      ! `shape` must have as many elements as `x`.
      x Got (4) and (3x3).

# prim_rev

    Code
      prim_rev(nv_array(1:4), axes = list(1L))
    Condition
      Error in `prim_rev()`:
      ! `axes` must be a whole number vector.
      x Got <list> of length 1.

# prim_cumsum

    Code
      prim_cumsum(nv_array(1:4), axis = c(1L, 1L))
    Condition
      Error in `prim_cumsum()`:
      ! `axis` must have length 1.
      x Got c(1, 1).

# prim_sum

    Code
      prim_sum(nv_array(1:4), axes = 1L, drop = "yes")
    Condition
      Error in `prim_sum()`:
      ! `drop` must be TRUE or FALSE.
      x Got "yes".

---

    Code
      prim_sum(nv_array(1:4), axes = 1L, drop = NA)
    Condition
      Error in `prim_sum()`:
      ! `drop` must be TRUE or FALSE.
      x Got NA.

---

    Code
      prim_sum(nv_array(1:4), axes = "a")
    Condition
      Error in `prim_sum()`:
      ! `axes` must be a whole number vector.
      x Got "a".

# prim_convert

    Code
      prim_convert(nv_array(1:4), dtype = "nope")
    Condition
      Error in `prim_convert()`:
      ! `dtype` must name a data type.
      x Got "nope".
      i See `tengen::as_dtype()` for the data types anvl knows.

---

    Code
      prim_convert(nv_array(1:4), dtype = 42)
    Condition
      Error in `prim_convert()`:
      ! `dtype` must name a data type.
      x Got 42.
      i See `tengen::as_dtype()` for the data types anvl knows.

# prim_round

    Code
      prim_round(nv_array(c(1.5, 2.5)), method = "bogus")
    Condition
      Error in `prim_round()`:
      ! `method` must be one of "nearest_even" or "afz".
      x Got "bogus".

# prim_rng_bit_generator

    Code
      prim_rng_bit_generator(state, "MERSENNE", "f32", 3L)
    Condition
      Error in `prim_rng_bit_generator()`:
      ! `rng_algorithm` must be one of "DEFAULT", "THREE_FRY", or "PHILOX".
      x Got "MERSENNE".

# prim_fill

    Code
      prim_fill(c(1, 2), 3L, "f32")
    Condition
      Error in `prim_fill()`:
      ! `value` must be a scalar.
      x Got c(1, 2).

---

    Code
      prim_fill(1, 3L, "nope")
    Condition
      Error in `prim_fill()`:
      ! `dtype` must name a data type.
      x Got "nope".
      i See `tengen::as_dtype()` for the data types anvl knows.

# prim_iota

    Code
      prim_iota(axis = 1L, dtype = "f32", shape = "a")
    Condition
      Error in `prim_iota()`:
      ! `shape` must be a whole number vector.
      x Got "a".

---

    Code
      prim_iota(axis = 1L, dtype = "bool", shape = 3L)
    Condition
      Error in `prim_iota()`:
      ! `dtype` must name an integer, unsigned integer, or float data type.
      x Got "bool".

# prim_broadcast_in_axes

    Code
      prim_broadcast_in_axes(x, shape = c(4L, 3L), broadcast_axes = 1L)
    Condition
      Error in `prim_broadcast_in_axes()`:
      ! `broadcast_axes` must have one entry per axis of `x`.
      x Got 1 for an `x` with 2 axes.

# prim_static_slice

    Code
      prim_static_slice(x, 1L, c(2L, 2L), 1L)
    Condition
      Error in `prim_static_slice()`:
      ! `start_indices`, `end_indices` and `strides` must have one entry per axis of `x` (2).
      x Got `start_indices` = 1, `end_indices` = c(2, 2), `strides` = 1.

---

    Code
      prim_static_slice(x, c(1L, 1L), c(2L, 2L), c(0L, 1L))
    Condition
      Error in `prim_static_slice()`:
      ! `strides` must be positive.
      x Got c(0, 1).

# prim_pad

    Code
      prim_pad(x, nv_scalar(0), 0L, c(0L, 0L), c(0L, 0L))
    Condition
      Error in `prim_pad()`:
      ! `edge_padding_low` must have one entry per axis of `x` (2).
      x Got 0.

---

    Code
      prim_pad(nv_array(as.double(1:3)), nv_scalar(0), -3L, -3L, 0L)
    Condition
      Error in `prim_pad()`:
      ! Negative padding must not remove more than an axis holds.
      x `x` has shape (3); axis 1 would end up at -3.
      i Got `edge_padding_low` = -3, `edge_padding_high` = -3, `interior_padding` = 0.

# prim_dynamic_slice

    Code
      prim_dynamic_slice(x, nv_scalar(1L), nv_scalar(1L), slice_sizes = 2L)
    Condition
      Error in `prim_dynamic_slice()`:
      ! `slice_sizes` must have one entry per axis of `x` (2).
      x Got 2.

# prim_top_k

    Code
      prim_top_k(nv_array(1:4), k = 1L, indices = "yes")
    Condition
      Error in `prim_top_k()`:
      ! `indices` must be TRUE or FALSE.
      x Got "yes".

---

    Code
      prim_top_k(nv_array(1:4), k = c(1L, 2L))
    Condition
      Error in `prim_top_k()`:
      ! `k` must have 1 entry.
      x Got c(1, 2).

# prim_chol

    Code
      prim_chol(m, lower = "x")
    Condition
      Error in `prim_chol()`:
      ! `lower` must be TRUE or FALSE.
      x Got "x".

# prim_sort

    Code
      prim_sort(list(), axis = 1L)
    Condition
      Error in `prim_sort()`:
      ! `xs` must be a non-empty list of arrayish values.
      x Got list().

---

    Code
      prim_sort(list(nv_array(1:4)), axis = 1L, decreasing = "yes")
    Condition
      Error in `prim_sort()`:
      ! `decreasing` must be TRUE or FALSE.
      x Got "yes".

# prim_dot_general

    Code
      prim_dot_general(lhs, rhs, contracting_axes = 1L, batching_axes = list(integer(),
      integer()))
    Condition
      Error in `prim_dot_general()`:
      ! `contracting_axes` must be a list of two axis vectors, one for `lhs` and one for `rhs`.
      x Got 1.

---

    Code
      prim_dot_general(lhs, rhs, contracting_axes = list(2L, 1L), batching_axes = list(
        1L, integer()))
    Condition
      Error in `prim_dot_general()`:
      ! `batching_axes` must name as many axes of `lhs` as of `rhs`.
      x Got 1 and integer(0).

---

    Code
      prim_dot_general(lhs, rhs, contracting_axes = list(c(1L, 2L), 1L),
      batching_axes = list(integer(), integer()))
    Condition
      Error in `prim_dot_general()`:
      ! `contracting_axes` must name as many axes of `lhs` as of `rhs`.
      x Got c(1, 2) and 1.

---

    Code
      prim_dot_general(lhs, rhs, contracting_axes = list(2L, 1L), batching_axes = list(
        integer(), integer()), precision = "bogus")
    Condition
      Error in `prim_dot_general()`:
      ! `precision` must be one of "default", "high", or "highest".
      x Got "bogus".

# prim_gather

    Code
      gather(slice_sizes = c(1L, 3L, 1L))
    Condition
      Error in `prim_gather()`:
      ! `slice_sizes` must have one entry per axis of `x` (2).
      x Got c(1, 3, 1).

---

    Code
      gather(start_index_map = c(1L, 2L))
    Condition
      Error in `prim_gather()`:
      ! `start_index_map` must have one entry per index coordinate (1).
      x Got c(1, 2).

---

    Code
      gather(collapsed_slice_axes = integer(), x_batching_axes = 1L)
    Condition
      Error in `prim_gather()`:
      ! `x_batching_axes` and `start_indices_batching_axes` must have the same length.
      x Got 1 and integer(0).

# prim_scatter

    Code
      scatter(scatter_axes_to_x_axes = c(1L, 2L))
    Condition
      Error in `prim_scatter()`:
      ! `scatter_axes_to_x_axes` must have one entry per index coordinate (1).
      x Got c(1, 2).

---

    Code
      scatter(inserted_window_axes = integer(), x_batching_axes = 1L)
    Condition
      Error in `prim_scatter()`:
      ! `x_batching_axes` and `scatter_indices_batching_axes` must have the same length.
      x Got 1 and integer(0).

# prim_convolution

    Code
      conv(window_strides = c(1L, 1L))
    Condition
      Error in `prim_convolution()`:
      ! `window_strides` must have one entry per spatial axis (1).
      x Got c(1, 1).

---

    Code
      conv(x_spatial_axes = c(3L, 4L))
    Condition
      Error in `prim_convolution()`:
      ! `x_spatial_axes` must have one entry per spatial axis (1).
      x Got c(3, 4).

---

    Code
      conv(x_batch_axis = 2L)
    Condition
      Error in `prim_convolution()`:
      ! The axes of x must each be named exactly once.
      x Axis 2 is named 2 times, by `x_batch_axis` and `x_feature_axis`.
      i Got `x_batch_axis` = 2, `x_spatial_axes` = 3, `x_feature_axis` = 2.

---

    Code
      conv(x_batch_axis = integer())
    Condition
      Error in `prim_convolution()`:
      ! The axes of x must each be named exactly once.
      x `x_batch_axis`, `x_spatial_axes`, and `x_feature_axis` name 2 axes between them, but x has 3.
      i Got `x_batch_axis` = integer(0), `x_spatial_axes` = 3, `x_feature_axis` = 2.

---

    Code
      conv(precision = "bogus")
    Condition
      Error in `prim_convolution()`:
      ! `precision` must be one of "default", "high", or "highest".
      x Got "bogus".

# value_repr() / spells a value the way format_param() does

    Code
      show_repr(NULL)
    Output
      NULL
    Code
      show_repr(3.5)
    Output
      3.5
    Code
      show_repr(TRUE)
    Output
      TRUE
    Code
      show_repr("afz")
    Output
      "afz"
    Code
      show_repr(c(1L, 3L))
    Output
      c(1, 3)
    Code
      show_repr(c("a", "b"))
    Output
      c("a", "b")
    Code
      show_repr(integer())
    Output
      integer(0)
    Code
      show_repr(character())
    Output
      character(0)

# value_repr() / cuts a long vector or string short

    Code
      show_repr(1:8)
    Output
      c(1, 2, 3, 4, 5, 6, 7, 8)
    Code
      show_repr(1:1000)
    Output
      c(1, 2, 3, 4, 5, 6, 7, 8, ...) of length 1000
    Code
      show_repr(rep("afz", 1000))
    Output
      c("afz", "afz", "afz", "afz", "afz", "afz", "afz", "afz", ...) of length 1000
    Code
      show_repr(strrep("a", 5000))
    Output
      "aaaaaaaaaaaaaaaaaaaaaaaaaaa..."

# value_repr() / copes with values format_param() never sees

    Code
      show_repr(NA)
    Output
      NA
    Code
      show_repr(NA_character_)
    Output
      NA
    Code
      show_repr(c(1L, NA))
    Output
      c(1, NA)
    Code
      show_repr(NaN)
    Output
      NaN
    Code
      show_repr(Inf)
    Output
      Inf
    Code
      show_repr(matrix(1:4, 2))
    Output
      matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
    Code
      show_repr(matrix(1:1000, 10))
    Output
      matrix(c(1, 2, 3, 4, 5, 6, 7, 8, ...), nrow = 10, ncol = 100)
    Code
      show_repr(matrix(5L, 1, 1))
    Output
      matrix(5, nrow = 1, ncol = 1)
    Code
      show_repr(matrix(integer(), 0, 3))
    Output
      matrix(integer(0), nrow = 0, ncol = 3)
    Code
      show_repr(array(1:8, c(2, 2, 2)))
    Output
      array(c(1, 2, 3, 4, 5, 6, 7, 8), dim = c(2, 2, 2))
    Code
      show_repr(array(1:3))
    Output
      array(c(1, 2, 3), dim = 3)
    Code
      show_repr(list())
    Output
      list()
    Code
      show_repr(as.list(1:1000))
    Output
      <list> of length 1000
    Code
      show_repr(factor(letters))
    Output
      <factor>
    Code
      show_repr(mean)
    Output
      <function>
    Code
      show_repr(globalenv())
    Output
      <environment>

# messages stay short for oversized params

    Code
      prim_fill(1:1000, 3L, "f32")
    Condition
      Error in `prim_fill()`:
      ! `value` must be a scalar.
      x Got c(1, 2, 3, 4, 5, 6, 7, 8, ...) of length 1000.

---

    Code
      prim_fill(1, 3L, strrep("a", 5000))
    Condition
      Error in `prim_fill()`:
      ! `dtype` must name a data type.
      x Got "aaaaaaaaaaaaaaaaaaaaaaaaaaa...".
      i See `tengen::as_dtype()` for the data types anvl knows.

---

    Code
      prim_round(nv_array(c(1.5, 2.5)), method = rep("afz", 1000))
    Condition
      Error in `prim_round()`:
      ! `method` must be one of "nearest_even" or "afz".
      x Got c("afz", "afz", "afz", "afz", "afz", "afz", "afz", "afz", ...) of length 1000.

---

    Code
      prim_static_slice(nv_array(1:4), 1:1000, 1:1000, 1:1000)
    Condition
      Error in `prim_static_slice()`:
      ! `start_indices`, `end_indices` and `strides` must have one entry per axis of `x` (1).
      x Got `start_indices` = c(1, 2, 3, 4, 5, 6, 7, 8, ...) of length 1000, `end_indices` = c(1, 2, 3, 4, 5, 6, 7, 8, ...) of length 1000, `strides` = c(1, 2, 3, 4, 5, 6, 7, 8, ...) of length 1000.

---

    Code
      prim_reshape(nv_array(1:4), shape = 1:1000)
    Condition
      Error in `prim_reshape()`:
      ! `shape` must have as many elements as `x`.
      x Got (4) and (1x2x3x4x5x6x7x8x...) with 1000 axes.

# a shape with too many elements is refused before it reaches XLA

    Code
      prim_fill(1, 1:1000, "f32")
    Condition
      Error in `prim_fill()`:
      ! `shape` must describe an array with fewer than 2^63 elements.
      x Got c(1, 2, 3, 4, 5, 6, 7, 8, ...) of length 1000.

# a whole-number param outside the integer range is not reported as NA

    Code
      prim_top_k(nv_array(1:4), k = Inf)
    Condition
      Error in `prim_top_k()`:
      ! `k` must be a whole number in the integer range.
      x Got Inf.

# broadcasting to a size-1 axis names 1 once

    Code
      prim_broadcast_in_axes(x, shape = c(1L, 3L), broadcast_axes = 1:2)
    Condition
      Error in `prim_broadcast_in_axes()`:
      ! Axis 1 of `x` must be 1 to broadcast to axis 1 of the result.
      x Got shapes (4x3) and (1x3).

# a result shape a rule computes from the caller's parameters / refuses a negative padding that empties a spatial axis past zero

    Code
      conv(padding = matrix(-100L, 2L, 2L))
    Condition
      Error in `prim_convolution()`:
      ! Negative `padding` must not remove more than spatial axis 1 of `x` holds.
      x Axis 3 of `x` dilates to 4, and padding -100 and -100 leaves -196.
      i Got `padding` = matrix(c(-100, -100, -100, -100), nrow = 2, ncol = 2), `x_dilation` = c(1, 1).

# a result shape a rule computes from the caller's parameters / refuses a zero-sized kernel spatial axis

    Code
      conv(kernel = nv_array(array(numeric(), c(1L, 1L, 0L, 0L)), dtype = "f32"))
    Condition
      Error in `prim_convolution()`:
      ! `kernel` must not have a zero-sized spatial axis.
      x Axis 3 of `kernel` is 0.

# a result shape a rule computes from the caller's parameters / refuses an overflowing dilation or padding rather than reaching an `if ()` with an `NA`

    Code
      conv(x_dilation = c(2000000000L, 1L))
    Condition
      Error in `prim_convolution()`:
      ! The convolution's result must have at most 2147483647 elements along each axis.
      x Axis 3 would end up at 6e+09.
      i Got `padding` = matrix(c(0, 0, 0, 0), nrow = 2, ncol = 2), `window_strides` = c(1, 1), `x_dilation` = c(2000000000, 1), `kernel_dilation` = c(1, 1).

---

    Code
      conv(padding = matrix(2000000000L, 2L, 2L))
    Condition
      Error in `prim_convolution()`:
      ! The convolution's result must have at most 2147483647 elements along each axis.
      x Axes c(3, 4) would end up at c(4e+09, 4e+09).
      i Got `padding` = matrix(c(2000000000, 2000000000, 2000000000, 2000000000), nrow = 2, ncol = 2), `window_strides` = c(1, 1), `x_dilation` = c(1, 1), `kernel_dilation` = c(1, 1).

# a result shape a rule computes from the caller's parameters / reports an out-of-range `end_indices` rather than overflowing on it

    Code
      prim_static_slice(nv_array(1:4), 1L, .Machine$integer.max, 1L)
    Condition
      Error in `prim_static_slice()`:
      ! `end_indices` must not exceed the shape of `x` (4).
      x Got 2147483647 at axis 1.

# a whole-number parameter the primitive fixes at one entry / is spoken of in the singular by every branch that refuses it

    Code
      prim_top_k(nv_array(1:4), NA)
    Condition
      Error in `prim_top_k()`:
      ! `k` must be a whole number.
      x Got NA.

---

    Code
      prim_top_k(nv_array(1:4), 2.5)
    Condition
      Error in `prim_top_k()`:
      ! `k` must be a whole number.
      x Got 2.5.

---

    Code
      prim_top_k(nv_array(1:4), 3e+09)
    Condition
      Error in `prim_top_k()`:
      ! `k` must be a whole number in the integer range.
      x Got 3e+09.

# the sub-graph arguments a primitive traces / refuses a `prim_if()` branch that is not a function

    Code
      prim_if(nv_scalar(TRUE), 1L, function() nv_scalar(1L))
    Condition
      Error in `prim_if()`:
      ! `true` must be a function.
      x Got 1.

---

    Code
      prim_if(nv_scalar(TRUE), function() nv_scalar(1L), "x")
    Condition
      Error in `prim_if()`:
      ! `false` must be a function.
      x Got "x".

