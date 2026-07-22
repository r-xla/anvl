test_that("integration: indexing-heavy graph matches PJRT", {
  skip_if_no_quickr_or_pjrt()

  operand <- array(sample.int(50, 24, replace = TRUE), dim = c(3L, 4L, 2L))
  idx <- matrix(c(0L, 4L), ncol = 1L) # gather clamps out-of-bounds
  padv <- 0L
  s1 <- 0L
  s2 <- 100L
  sc_idx <- matrix(c(2L, 6L), ncol = 1L)

  fn <- function(operand, idx, padv, s1, s2, sc_idx) {
    g <- prim_gather(
      operand,
      idx,
      slice_sizes = c(1L, 2L, 2L),
      offset_axes = c(2L, 3L),
      collapsed_slice_axes = 1L,
      operand_batching_axes = integer(),
      start_indices_batching_axes = integer(),
      start_index_map = 1L,
      index_vector_axis = 2L
    )

    r <- nv_reverse(g, axes = 3L)

    p <- prim_pad(
      r,
      padv,
      edge_padding_low = c(1L, 0L, 0L),
      edge_padding_high = c(0L, 1L, 0L),
      interior_padding = c(0L, 1L, 0L)
    )

    s <- prim_dynamic_slice(p, s1, s2, 1L, slice_sizes = c(2L, 2L, 2L))
    m <- prim_reshape(s, shape = c(2L, 4L))
    upd <- prim_reduce_sum(m, axes = 2L, drop = TRUE)

    base <- nv_iota(axis = 1L, dtype = "i32", shape = 6L, start = 0L)
    scattered <- prim_scatter(
      base,
      sc_idx,
      upd,
      update_window_axes = integer(),
      inserted_window_axes = 1L,
      input_batching_axes = integer(),
      scatter_indices_batching_axes = integer(),
      scatter_axes_to_operand_axes = 1L,
      index_vector_axis = 2L,
      update_computation = function(old, new) old + new
    )

    out_f64 <- prim_convert(scattered, dtype = "f64")
    list(slice = s, scattered = scattered, total = sum(out_f64))
  }

  templates <- list(
    operand = nv_array(operand, dtype = "i32", shape = dim(operand)),
    idx = nv_array(idx, dtype = "i32", shape = dim(idx)),
    padv = nv_scalar(0L, dtype = "i32"),
    s1 = nv_scalar(0L, dtype = "i32"),
    s2 = nv_scalar(0L, dtype = "i32"),
    sc_idx = nv_array(sc_idx, dtype = "i32", shape = dim(sc_idx))
  )

  run <- list(
    args = list(
      operand = operand,
      idx = idx,
      padv = padv,
      s1 = s1,
      s2 = s2,
      sc_idx = sc_idx
    ),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(fn, templates, list(run))
})
