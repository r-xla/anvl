describe("nv_reshape shape inference", {
  it("infers a trailing dimension", {
    expect_equal(shape(nv_reshape(nv_array(1:6), c(2, -1))), c(2L, 3L))
  })
  it("infers a leading dimension", {
    expect_equal(shape(nv_reshape(nv_array(1:6), c(-1, 3))), c(2L, 3L))
  })
  it("flattens with a lone -1", {
    x <- nv_array(1:6, shape = c(2, 3))
    expect_equal(nv_reshape(x, -1), nv_flatten(x))
  })
  it("infers a middle dimension", {
    expect_equal(shape(nv_reshape(nv_array(1:12), c(2, -1, 2))), c(2L, 3L, 2L))
  })
  it("infers size 1 for a scalar", {
    expect_equal(shape(nv_reshape(nv_scalar(1), -1)), 1L)
  })
  it("rejects more than one -1", {
    expect_error(nv_reshape(nv_array(1:6), c(-1, -1)), "at most one")
  })
  it("rejects a shape that does not divide evenly", {
    expect_error(nv_reshape(nv_array(1:6), c(4, -1)), "Cannot infer dimension")
  })
  it("rejects negative values other than -1", {
    expect_error(nv_reshape(nv_array(1:6), c(2, -2)), "must contain only non-negative")
  })
})

describe("negative dimension arguments", {
  m <- nv_matrix(as.numeric(1:6), nrow = 2)
  b <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE), nrow = 2)
  x3 <- nv_array(as.numeric(1:24), shape = c(2, 3, 4))

  it("nv_transpose", {
    expect_equal(nv_transpose(x3, c(-1L, -2L, -3L)), nv_transpose(x3, c(3L, 2L, 1L)))
  })
  it("nv_concatenate", {
    expect_equal(nv_concatenate(m, m, dimension = -1L), nv_concatenate(m, m, dimension = 2L))
  })
  it("nv_reverse", {
    expect_equal(nv_reverse(x3, dims = c(-1L, -3L)), nv_reverse(x3, dims = c(3L, 1L)))
  })
  it("nv_iota", {
    expect_equal(
      nv_iota(dim = -1L, dtype = "i32", shape = c(2, 3)),
      nv_iota(dim = 2L, dtype = "i32", shape = c(2, 3))
    )
  })
  it("nv_iota_like", {
    expect_equal(nv_iota_like(m, dim = -1L), nv_iota_like(m, dim = 2L))
  })
  it("nv_reduce_sum", {
    expect_equal(nv_reduce_sum(x3, dims = -1L), nv_reduce_sum(x3, dims = 3L))
  })
  it("nv_reduce_prod", {
    expect_equal(nv_reduce_prod(x3, dims = -2L), nv_reduce_prod(x3, dims = 2L))
  })
  it("nv_reduce_max", {
    expect_equal(nv_reduce_max(x3, dims = -1L), nv_reduce_max(x3, dims = 3L))
  })
  it("nv_reduce_min", {
    expect_equal(nv_reduce_min(x3, dims = -1L), nv_reduce_min(x3, dims = 3L))
  })
  it("nv_reduce_any", {
    expect_equal(nv_reduce_any(b, dims = -1L), nv_reduce_any(b, dims = 2L))
  })
  it("nv_reduce_all", {
    expect_equal(nv_reduce_all(b, dims = -1L), nv_reduce_all(b, dims = 2L))
  })
  it("nv_mean", {
    expect_equal(nv_mean(x3, dims = -1L), nv_mean(x3, dims = 3L))
  })
  it("nv_var", {
    expect_equal(nv_var(m, dims = -1L), nv_var(m, dims = 2L))
  })
  it("nv_sd", {
    expect_equal(nv_sd(m, dims = -1L), nv_sd(m, dims = 2L))
  })
  it("nv_cumsum", {
    expect_equal(nv_cumsum(m, dim = -1L), nv_cumsum(m, dim = 2L))
  })
  it("nv_cumprod", {
    expect_equal(nv_cumprod(m, dim = -2L), nv_cumprod(m, dim = 1L))
  })
  it("nv_cummax", {
    expect_equal(nv_cummax(m, dim = -1L), nv_cummax(m, dim = 2L))
  })
  it("nv_cummin", {
    expect_equal(nv_cummin(m, dim = -1L), nv_cummin(m, dim = 2L))
  })
  it("nv_squeeze", {
    x <- nv_array(1:6, shape = c(1, 6, 1))
    expect_equal(nv_squeeze(x, dims = -1L), nv_squeeze(x, dims = 3L))
  })
  it("nv_unsqueeze counts from the end of the result", {
    x <- nv_array(c(1, 2, 3))
    expect_equal(shape(nv_unsqueeze(x, dim = -1L)), c(3L, 1L))
    expect_equal(shape(nv_unsqueeze(x, dim = -2L)), c(1L, 3L))
  })
  it("nv_select", {
    expect_equal(nv_select(m, dim = -1L, index = 2L), nv_select(m, dim = 2L, index = 2L))
  })
  it("nv_sort", {
    expect_equal(nv_sort(m, dim = -2L), nv_sort(m, dim = 1L))
  })
  it("nv_argsort", {
    expect_equal(nv_argsort(m, dim = -2L), nv_argsort(m, dim = 1L))
  })
  it("nv_top_k", {
    expect_equal(nv_top_k(m, k = 2L, dim = -1L), nv_top_k(m, k = 2L, dim = 2L))
  })
  it("nv_quantile", {
    expect_equal(nv_quantile(m, 0.5, dim = -1L), nv_quantile(m, 0.5, dim = 2L))
  })
  it("nv_median", {
    expect_equal(nv_median(m, dim = -1L), nv_median(m, dim = 2L))
  })
  it("nv_argmax", {
    expect_equal(nv_argmax(m, dim = -1L), nv_argmax(m, dim = 2L))
  })
  it("nv_argmin", {
    expect_equal(nv_argmin(m, dim = -1L), nv_argmin(m, dim = 2L))
  })
})

describe("dimension argument validation", {
  m <- nv_matrix(as.numeric(1:6), nrow = 2)

  it("rejects out-of-range negative dims", {
    expect_error(nv_reduce_sum(m, dims = -3L), "between 1 and 2, or between -2 and -1")
  })
  it("rejects out-of-range positive dims", {
    expect_error(nv_cumsum(m, dim = 3L), "between 1 and 2, or between -2 and -1")
  })
  it("rejects 0", {
    expect_error(nv_sort(m, dim = 0L), "between 1 and 2, or between -2 and -1")
  })
  it("rejects mixing a positive and a negative reference to the same dim", {
    expect_error(nv_reduce_sum(m, dims = c(2L, -1L)), "duplicate dimensions")
  })
  it("rejects a non-scalar dim", {
    expect_error(nv_cumsum(m, dim = c(1L, 2L)), "must have length 1")
  })
  it("allows unsqueeze one past the end", {
    expect_equal(shape(nv_unsqueeze(m, dim = 3L)), c(2L, 3L, 1L))
    expect_error(nv_unsqueeze(m, dim = 4L), "between 1 and 3")
  })
})
