test_that("shape2string", {
  expect_equal(shape2string(c(2, 2)), "(2,2)")
  expect_equal(shape2string(c(2, 2), parenthesize = FALSE), "2,2")
  expect_equal(shape2string(Shape(c(2, 2))), "(2,2)")
  expect_equal(shape2string(Shape(c()), parenthesize = TRUE), "()")
  expect_equal(shape2string(Shape(c()), parenthesize = FALSE), "")
})

test_that("peek_dtype", {
  expect_equal(
    peek_dtype(1L),
    as_dtype("i32")
  )
  expect_equal(
    peek_dtype(nv_scalar(1L, dtype = "f32")),
    as_dtype("f32")
  )
})

test_that("naxes_abstract", {
  expect_equal(naxes_abstract(1L), 0L)
  expect_equal(naxes_abstract(nv_array(1:4, dtype = "f32", shape = c(2, 2))), 2L)
})

test_that("shape_abstract", {
  expect_equal(shape_abstract(1L), integer())
  expect_equal(shape_abstract(nv_array(1:4, dtype = "f32", shape = c(2, 2))), c(2, 2))
})

describe("gather_clamp_indices", {
  it("clamps indices that exceed upper bound (implicit index vector)", {
    # `x` axis=10, slice_size=3, so max valid start = 10-3+1 = 8
    idx <- nv_array(c(9L, 10L, 5L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      x_shape = c(10L),
      slice_sizes = 3L,
      start_index_map = 1L,
      index_vector_axis = 2L # implicit
    )
    expect_equal(as.integer(result), c(8L, 8L, 5L))
  })

  it("clamps indices below 1 to 1 (implicit index vector)", {
    idx <- nv_array(c(0L, -2L, 3L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      x_shape = c(10L),
      slice_sizes = 1L,
      start_index_map = 1L,
      index_vector_axis = 2L
    )
    expect_equal(as.integer(result), c(1L, 1L, 3L))
  })

  it("leaves valid indices unchanged (implicit index vector)", {
    idx <- nv_array(c(1L, 5L, 8L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      x_shape = c(10L),
      slice_sizes = c(3L),
      start_index_map = 1L,
      index_vector_axis = 2L
    )
    expect_equal(as.integer(result), c(1L, 5L, 8L))
  })

  it("clamps with explicit index vector axis (multiple coordinates)", {
    # Shape [2]: two coordinates, one per axis of `x`
    # x_shape = c(5, 8), slice_sizes = c(2, 3)
    # max for axis1 = 5-2+1 = 4, max for axis2 = 8-3+1 = 6
    idx <- nv_array(c(10L, 10L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      x_shape = c(5L, 8L),
      slice_sizes = c(2L, 3L),
      start_index_map = c(1L, 2L),
      index_vector_axis = 1L
    )
    expect_equal(as.integer(result), c(4L, 6L))
  })

  it("clamps batch of indices with explicit index vector axis and reverse start_index_map", {
    # x_shape = c(8, 5), slice_sizes = c(2, 3)
    # because we reverse the start_index_map, we clamp:
    # clamp(1, coord_1, max(1, 5 - 3 + 1) = 3)
    # clamp(1, coord_2, max(1, 8 - 2 + 1) = 7)
    idx <- nv_array(
      matrix(
        c(
          7L,
          0L,
          4L,
          3L,
          7L,
          10L
        ),
        nrow = 3,
        byrow = TRUE
      ),
      dtype = "i32"
    )
    result <- gather_clamp_indices(
      start_indices = idx,
      x_shape = c(8L, 5L),
      slice_sizes = c(2L, 3L),
      start_index_map = c(2L, 1L),
      index_vector_axis = 2L
    )
    expected <- matrix(
      c(
        3L,
        1L,
        3L,
        3L,
        3L,
        7L
      ),
      nrow = 3,
      byrow = TRUE
    )
    expect_equal(as_array(result), expected)
  })

  it("handles slice_size equal to axis size (max_bound = 1)", {
    # slice covers the whole axis, so only valid start is 1
    idx <- nv_array(c(0L, 5L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      x_shape = 10L,
      slice_sizes = 10L,
      start_index_map = 1L,
      index_vector_axis = 2L
    )
    expect_equal(nv_array(c(1L, 1L), dtype = "i32"), result)
  })
})

describe("scatter_to_gather_slice_sizes", {
  it("x[2:5] on 1D array: range is a window axis", {
    # update_shape = c(4), one window axis covering the range
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(4L),
      x_shape = c(10L),
      update_window_axes = 1L,
      inserted_window_axes = integer(),
      x_batching_axes = integer()
    )
    expect_equal(result, 4L)
  })

  it("x[3, ] on 2D array: scalar drops axis 1, axis 2 is window", {
    # update_shape = c(5), inserted axis 1, window axis 2
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(5L),
      x_shape = c(4L, 5L),
      update_window_axes = 1L,
      inserted_window_axes = 1L,
      x_batching_axes = integer()
    )
    expect_equal(result, c(1L, 5L))
  })

  it("x[, 2] on 2D array: axis 1 is window, scalar drops axis 2", {
    # update_shape = c(4), inserted axis 2, window axis 1
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(4L),
      x_shape = c(4L, 5L),
      update_window_axes = 1L,
      inserted_window_axes = 2L,
      x_batching_axes = integer()
    )
    expect_equal(result, c(4L, 1L))
  })

  it("x[1:2, 1:3] on 2D array: both axes are windows", {
    # update_shape = c(2, 3), both axes are windows
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(2L, 3L),
      x_shape = c(4L, 5L),
      update_window_axes = c(1L, 2L),
      inserted_window_axes = integer(),
      x_batching_axes = integer()
    )
    expect_equal(result, c(2L, 3L))
  })

  it("x[2, 3] on 2D array: both axes inserted (scalar update)", {
    result <- scatter_to_gather_slice_sizes(
      update_shape = integer(),
      x_shape = c(4L, 5L),
      update_window_axes = integer(),
      inserted_window_axes = c(1L, 2L),
      x_batching_axes = integer()
    )
    expect_equal(result, c(1L, 1L))
  })

  it("x[2, 1:3, ] on 3D array: axis 1 inserted, axes 2-3 windows", {
    # update_shape = c(3, 6)
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(3L, 6L),
      x_shape = c(4L, 5L, 6L),
      update_window_axes = c(1L, 2L),
      inserted_window_axes = 1L,
      x_batching_axes = integer()
    )
    expect_equal(result, c(1L, 3L, 6L))
  })
})
