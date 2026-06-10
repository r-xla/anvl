test_that("shape2string", {
  expect_equal(shape2string(c(2, 2)), "(2,2)")
  expect_equal(shape2string(c(2, 2), parenthesize = FALSE), "2,2")
  expect_equal(shape2string(Shape(c(2, 2))), "(2,2)")
  expect_equal(shape2string(Shape(c()), parenthesize = TRUE), "()")
  expect_equal(shape2string(Shape(c()), parenthesize = FALSE), "")
})

test_that("dtype2string", {
  expect_equal(dtype2string(as_dtype("f32")), "f32")
  expect_equal(dtype2string(as_dtype("i32")), "i32")
  expect_equal(dtype2string(as_dtype("f32"), ambiguous = TRUE), "f32?")
})

test_that("dtype_abstract", {
  expect_equal(
    dtype_abstract(1L),
    as_dtype("i32")
  )
  expect_equal(
    dtype_abstract(nv_scalar(1L, dtype = "f32")),
    as_dtype("f32")
  )
})

test_that("ndims_abstract", {
  expect_equal(ndims_abstract(1L), 0L)
  expect_equal(ndims_abstract(nv_array(1:4, dtype = "f32", shape = c(2, 2))), 2L)
})

test_that("shape_abstract", {
  expect_equal(shape_abstract(1L), integer())
  expect_equal(shape_abstract(nv_array(1:4, dtype = "f32", shape = c(2, 2))), c(2, 2))
})

describe("gather_clamp_indices", {
  it("clamps indices that exceed upper bound (implicit index vector)", {
    # operand dim=10, slice_size=3, so max valid start = 10-3+1 = 8
    idx <- nv_array(c(9L, 10L, 5L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      operand_shape = c(10L),
      slice_sizes = 3L,
      start_index_map = 1L,
      index_vector_dim = 2L # implicit
    )
    expect_equal(as.integer(result), c(8L, 8L, 5L))
  })

  it("clamps indices below 1 to 1 (implicit index vector)", {
    idx <- nv_array(c(0L, -2L, 3L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      operand_shape = c(10L),
      slice_sizes = 1L,
      start_index_map = 1L,
      index_vector_dim = 2L
    )
    expect_equal(as.integer(result), c(1L, 1L, 3L))
  })

  it("leaves valid indices unchanged (implicit index vector)", {
    idx <- nv_array(c(1L, 5L, 8L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      operand_shape = c(10L),
      slice_sizes = c(3L),
      start_index_map = 1L,
      index_vector_dim = 2L
    )
    expect_equal(as.integer(result), c(1L, 5L, 8L))
  })

  it("clamps with explicit index vector dim (multiple coordinates)", {
    # Shape [2]: two coordinates, one per operand dim
    # operand_shape = c(5, 8), slice_sizes = c(2, 3)
    # max for dim1 = 5-2+1 = 4, max for dim2 = 8-3+1 = 6
    idx <- nv_array(c(10L, 10L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      operand_shape = c(5L, 8L),
      slice_sizes = c(2L, 3L),
      start_index_map = c(1L, 2L),
      index_vector_dim = 1L
    )
    expect_equal(as.integer(result), c(4L, 6L))
  })

  it("clamps batch of indices with explicit index vector dim and reverse start_index_map", {
    # operand_shape = c(8, 5), slice_sizes = c(2, 3)
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
      operand_shape = c(8L, 5L),
      slice_sizes = c(2L, 3L),
      start_index_map = c(2L, 1L),
      index_vector_dim = 2L
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

  it("handles slice_size equal to dim size (max_bound = 1)", {
    # slice covers the whole dim, so only valid start is 1
    idx <- nv_array(c(0L, 5L), dtype = "i32")
    result <- gather_clamp_indices(
      start_indices = idx,
      operand_shape = 10L,
      slice_sizes = 10L,
      start_index_map = 1L,
      index_vector_dim = 2L
    )
    expect_equal(nv_array(c(1L, 1L), dtype = "i32"), result)
  })
})

test_that("ambiguous_abstract", {
  # Non-ambiguous scalar (no explicit dtype)
  expect_false(ambiguous_abstract(nv_scalar(1.0)))
  expect_false(ambiguous_abstract(nv_scalar(1L)))

  # Non-ambiguous scalar (explicit dtype)
  expect_false(ambiguous_abstract(nv_scalar(1.0, dtype = "f32")))
  expect_false(ambiguous_abstract(nv_scalar(1L, dtype = "i32")))

  # Non-ambiguous array
  expect_false(ambiguous_abstract(nv_array(1:4, dtype = "f32", shape = c(2, 2))))

  # Primitive R values (converted via to_abstract)
  expect_true(ambiguous_abstract(1.0))
  expect_true(ambiguous_abstract(1L))
  expect_false(ambiguous_abstract(TRUE))
})

describe("scatter_to_gather_slice_sizes", {
  it("x[2:5] on 1D array: range is a window dim", {
    # update_shape = c(4), one window dim covering the range
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(4L),
      input_shape = c(10L),
      update_window_dims = 1L,
      inserted_window_dims = integer(),
      input_batching_dims = integer()
    )
    expect_equal(result, 4L)
  })

  it("x[3, ] on 2D array: scalar drops dim 1, dim 2 is window", {
    # update_shape = c(5), inserted dim 1, window dim 2
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(5L),
      input_shape = c(4L, 5L),
      update_window_dims = 1L,
      inserted_window_dims = 1L,
      input_batching_dims = integer()
    )
    expect_equal(result, c(1L, 5L))
  })

  it("x[, 2] on 2D array: dim 1 is window, scalar drops dim 2", {
    # update_shape = c(4), inserted dim 2, window dim 1
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(4L),
      input_shape = c(4L, 5L),
      update_window_dims = 1L,
      inserted_window_dims = 2L,
      input_batching_dims = integer()
    )
    expect_equal(result, c(4L, 1L))
  })

  it("x[1:2, 1:3] on 2D array: both dims are windows", {
    # update_shape = c(2, 3), both dims are windows
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(2L, 3L),
      input_shape = c(4L, 5L),
      update_window_dims = c(1L, 2L),
      inserted_window_dims = integer(),
      input_batching_dims = integer()
    )
    expect_equal(result, c(2L, 3L))
  })

  it("x[2, 3] on 2D array: both dims inserted (scalar update)", {
    result <- scatter_to_gather_slice_sizes(
      update_shape = integer(),
      input_shape = c(4L, 5L),
      update_window_dims = integer(),
      inserted_window_dims = c(1L, 2L),
      input_batching_dims = integer()
    )
    expect_equal(result, c(1L, 1L))
  })

  it("x[2, 1:3, ] on 3D array: dim 1 inserted, dims 2-3 windows", {
    # update_shape = c(3, 6)
    result <- scatter_to_gather_slice_sizes(
      update_shape = c(3L, 6L),
      input_shape = c(4L, 5L, 6L),
      update_window_dims = c(1L, 2L),
      inserted_window_dims = 1L,
      input_batching_dims = integer()
    )
    expect_equal(result, c(1L, 3L, 6L))
  })
})
