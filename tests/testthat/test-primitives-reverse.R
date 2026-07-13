# most reverse tests are in extra-tests for comparison with torch
# Just a few selected ones that don't use torch are here

test_that("prim_dot_general: vector dot product gradient", {
  # y = <x, y> = sum_i x_i * y_i, scalar output
  f <- function(x, y) {
    prim_dot_general(
      x,
      y,
      contracting_dims = list(1L, 1L),
      batching_dims = list(integer(), integer())
    )
  }
  g <- jit(gradient(f))
  x <- nv_array(c(1, 2, 3), dtype = "f32", shape = 3L)
  y <- nv_array(c(4, 5, 6), dtype = "f32", shape = 3L)
  # d/dx = y; d/dy = x
  out <- g(x, y)
  expect_equal(as.numeric(pjrt::as_array(out[[1L]])), c(4, 5, 6))
  expect_equal(as.numeric(pjrt::as_array(out[[2L]])), c(1, 2, 3))
})

test_that("prim_dot_general: matrix-vector with summed loss", {
  nv_hacky_sum <- function(y) {
    ones <- nv_array(1, shape = shape(y)[1L], dtype = dtype(y))
    out <- prim_dot_general(
      y,
      ones,
      contracting_dims = list(1L, 1L),
      batching_dims = list(integer(), integer())
    )
    return(out)
  }

  f <- function(A, x) {
    y <- prim_dot_general(
      A,
      x,
      contracting_dims = list(2L, 1L),
      batching_dims = list(integer(), integer())
    )
    nv_hacky_sum(y)
  }
  g <- jit(gradient(f))
  A <- nv_array(
    matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2),
    dtype = "f32",
    shape = c(2L, 2L)
  )
  x <- nv_array(c(5, 6), dtype = "f32", shape = 2L)
  jit(f)(A, x)
  out <- g(A, x)
  dA <- pjrt::as_array(out[[1L]])
  dx <- pjrt::as_array(out[[2L]])

  # dy/dA = [1, 1]^T * x
  dA_true <- outer(c(1, 1), as_array(x))
  # dy/dx = A^T * [1, 1]
  dx_true <- array(t(as_array(A)) %*% c(1, 1), dim = 2L)
  expect_equal(dA, dA_true)
  expect_equal(dx, dx_true)
})

test_that("prim_dot_general: batched matmul gradient w.r.t both inputs", {
  # Helpers to reduce repetition
  make_ones_like <- function(Y) {
    nv_array(
      rep(1, prod(shape(Y))),
      # TODO: repr() should not be needed
      dtype = dtype(Y),
      shape = shape(Y)
    )
  }
  sum_all <- function(Y) {
    ones <- make_ones_like(Y)
    rank <- length(shape(Y))
    all_dims <- 1:rank
    prim_dot_general(
      Y,
      ones,
      contracting_dims = list(all_dims, all_dims),
      batching_dims = list(integer(), integer())
    )
  }
  check_case <- function(A, B, contracting_dims, batching_dims) {
    f <- function(A, B) {
      prim_dot_general(
        A,
        B,
        contracting_dims = contracting_dims,
        batching_dims = batching_dims
      )
    }
    l <- function(A, B) {
      sum_all(f(A, B))
    }
    g <- jit(gradient(l))
    out <- g(A, B)
    dA <- out[[1L]]
    dB <- out[[2L]]

    expect_equal(shape(dA), shape(A))
    expect_equal(shape(dB), shape(B))

    # Verify linearization: <A, dA> == l(A,B) and <B, dB> == l(A,B)
    all_dims_A <- seq_along(shape(A))
    lin_A <- function(A) {
      prim_dot_general(
        A,
        dA,
        contracting_dims = list(all_dims_A, all_dims_A),
        batching_dims = list(integer(), integer())
      )
    }
    all_dims_B <- seq_along(shape(B))
    lin_B <- function(B) {
      prim_dot_general(
        B,
        dB,
        contracting_dims = list(all_dims_B, all_dims_B),
        batching_dims = list(integer(), integer())
      )
    }
    expect_equal(lin_A(A), l(A, B))
    expect_equal(lin_B(B), l(A, B))
  }

  # Case 1: Single contracting dim, no batching dims (existing baseline)
  # A[b,m,k] • B[b,k,n] with contracting k only
  A1 <- nv_array(
    1:12,
    shape = c(2, 2, 3),
    dtype = "f32",
  )
  B1 <- nv_array(
    1:12,
    shape = c(2, 3, 2),
    dtype = "f32"
  )
  check_case(
    A1,
    B1,
    contracting_dims = list(3L, 2L),
    batching_dims = list(integer(), integer())
  )

  # Case 2: Multiple contracting dims, no batching dims
  # A[b,m,k1,k2] • B[b,k1,k2,n] with contracting (k1,k2)
  A2 <- nv_array(
    1:(2 * 2 * 2 * 3),
    shape = c(2, 2, 2, 3),
    dtype = "f32"
  )
  B2 <- nv_array(
    1:(2 * 2 * 3 * 2),
    shape = c(2, 3, 2, 2),
    dtype = "f32"
  )
  check_case(
    A2,
    B2,
    contracting_dims = list(c(3L, 4L), c(3L, 2L)),
    batching_dims = list(integer(), integer())
  )

  # Case 3: Multiple batching dims (b1,b2), single contracting dim
  # A[b1,b2,m,k] • B[b1,b2,k,n] with batching (b1,b2) and contracting k
  A3 <- nv_array(
    1:(1 * 6 * 3 * 2 * 3),
    shape = c(1, 6, 3, 2, 3),
    dtype = "f32"
  )
  B3 <- nv_array(
    1:(6 * 1 * 3 * 2),
    shape = c(6, 1, 3, 2),
    dtype = "f32"
  )
  check_case(
    A3,
    B3,
    contracting_dims = list(5L, 3L),
    batching_dims = list(c(1L, 2L), c(2L, 1L))
  )

  # Case 4: Multiple batching dims and multiple contracting dims
  # A[b1,b2,m,k1,k2] • B[b1,b2,k1,k2,n] with batching (b1,b2) and contracting (k1,k2)
  A4 <- nv_array(
    1:(2 * 3 * 2 * 2 * 3),
    shape = c(2, 3, 2, 2, 3),
    dtype = "f32"
  )
  B4 <- nv_array(
    1:(2 * 3 * 2 * 3 * 5),
    shape = c(2, 3, 2, 3, 5),
    dtype = "f32"
  )
  check_case(
    A4,
    B4,
    contracting_dims = list(c(4L, 5L), c(3L, 4L)),
    batching_dims = list(c(1L, 2L), c(1L, 2L))
  )

  # batching dims come at last
  A5 <- nv_array(
    1:(2 * 3 * 2 * 2 * 3),
    shape = c(2, 3, 2, 2, 3),
    dtype = "f32"
  )
  B5 <- nv_array(
    1:(2 * 3 * 2 * 3 * 5),
    shape = c(2, 3, 2, 3, 5),
    dtype = "f32"
  )
  check_case(
    A5,
    B5,
    contracting_dims = list(c(1L, 2L), c(3L, 4L)),
    batching_dims = list(c(4L, 5L), c(1L, 2L))
  )
})

test_that("broadcasting", {
  f <- jit(gradient(
    function(x, y) {
      mean(x + y)
    },
    wrt = c("x", "y")
  ))

  out <- f(nv_scalar(1), nv_array(0, shape = c(1, 2)))
  expect_equal(out[[1L]], nv_scalar(1))
  expect_equal(out[[2L]], nv_array(0.5, shape = c(1, 2)))
})

test_that("prim_if", {
  # TODO:
  #f <- jit(gradient(
  #  function(pred, x) {
  #    prim_if(pred, \() x * nv_scalar(1), \() x * nv_scalar(2))
  #  },
  #  wrt = "x"
  #))
  #out <- f(nv_scalar(TRUE), nv_scalar(2))
  #expect_equal(out[[1L]], nv_scalar(2))
  #expect_equal(out[[2L]], nv_scalar(1))
})

test_that("prim_log reverse", {
  f <- jit(gradient(function(x) {
    nv_log(x)
  }))

  x <- nv_scalar(2, dtype = "f32")
  grad <- f(x)[[1L]]

  expect_equal(as_array(grad), 0.5)
})

test_that("prim_exp", {
  f <- jit(gradient(function(x) {
    nv_exp(x)
  }))

  x <- nv_scalar(2, dtype = "f32")
  grad <- f(x)[[1L]]

  expect_equal(as_array(grad), exp(2))
})

test_that("prim_reduce_max reverse", {
  f <- jit(gradient(function(x) {
    rows_max <- prim_reduce_max(x, dims = 2L, drop = TRUE)
    nv_reduce_sum(rows_max, dims = 1L, drop = TRUE)
  }))

  x <- nv_array(
    rbind(
      c(1, 3, 2),
      c(2, 4, 5)
    )
  )

  grads <- f(x)[[1L]]

  expect_equal(
    as_array(grads),
    rbind(
      c(0, 1, 0),
      c(0, 0, 1)
    )
  )
})

test_that("prim_reduce_min reverse", {
  f <- jit(gradient(function(x) {
    rows_min <- prim_reduce_min(x, dims = 2L, drop = TRUE)
    nv_reduce_sum(rows_min, dims = 1L, drop = TRUE)
  }))

  x <- nv_array(
    rbind(
      c(2, 4, 5),
      c(1, 1, 9)
    ),
    dtype = "f32",
    shape = c(2, 3)
  )

  grads <- f(x)[[1L]]

  expect_equal(
    as_array(grads),
    rbind(
      c(1, 0, 0),
      c(0.5, 0.5, 0)
    ),
    tolerance = 1e-6
  )
})

test_that("prim_max on ties", {
  x <- nv_array(c(1, 2, 2))
  grads <- jit(gradient(\(x) nv_reduce_max(x, dims = 1)))(x)
  expect_equal(as_array(grads$x), array(c(0, 0.5, 0.5), dim = 3))
})

test_that("prim_max", {
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(3, 2, 1))

  grads <- jit(gradient(\(x, y) nv_reduce_sum(nv_max(x, y), dims = 1)))(x, y)

  expect_equal(as_array(grads$x), array(c(0, 0.5, 1), dim = 3))
  expect_equal(as_array(grads$y), array(c(1, 0.5, 0), dim = 3))
})

test_that("prim_min", {
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(3, 2, 1))

  grads <- jit(gradient(\(x, y) nv_reduce_sum(nv_min(x, y), dims = 1)))(x, y)

  expect_equal(as_array(grads$x), array(c(1, 0.5, 0), dim = 3))
  expect_equal(as_array(grads$y), array(c(0, 0.5, 1), dim = 3))
})

test_that("prim_convert reverse converts gradients to the input dtype", {
  x_arr <- array(1:6, c(2, 3))
  x <- nv_array(x_arr, dtype = "f32")
  f <- jit(gradient(function(x) {
    y <- prim_convert(x, dtype = "f64", ambiguous = FALSE)
    nv_reduce_sum(y, dims = 1:2, drop = TRUE)
  }))

  grads <- f(x)
  expect_equal(as_array(grads[[1L]]), array(1, dim = dim(x_arr)))
  expect_equal(dtype(grads[[1L]]), as_dtype("f32"))
})

test_that("prim_eq, prim_ne, prim_gt, prim_ge, prim_lt, prim_le", {
  a <- 1
  b <- 2

  a_nv <- nv_scalar(a)
  b_nv <- nv_scalar(b)

  comparators <- list(
    list(fun = nv_eq, expected = a == b),
    list(fun = nv_ne, expected = a != b),
    list(fun = nv_gt, expected = a > b),
    list(fun = nv_ge, expected = a >= b),
    list(fun = nv_lt, expected = a < b),
    list(fun = nv_le, expected = a <= b)
  )

  for (cmp in comparators) {
    out <- as_array(cmp$fun(a_nv, b_nv))
    expect_identical(out, cmp$expected)

    g <- jit(gradient(function(a, b) {
      cmp$fun(a, b)
    }))

    # can't compute gradients if function doesn't return
    # float
    expect_snapshot_error({
      grads <- g(a_nv, b_nv)
    })

    g <- jit(gradient(function(a, b) {
      nv_convert(cmp$fun(a, b), "f32")
    }))

    grads <- g(a_nv, b_nv)
    expect_equal(as_array(grads[[1L]]), 0)
    expect_equal(as_array(grads[[2L]]), 0)
  }
})

test_that("prim_pad reverse with interior padding", {
  # Interior padding adds padding between elements
  # For input [a, b, c] with interior_padding=1, output is [a, 0, b, 0, c]
  # Gradient flows only to original positions [a, b, c]
  f <- jit(gradient(function(x) {
    x <- x * nv_array(c(1, 2, 3), dtype = "f64")
    y <- prim_pad(x, nv_scalar(0, "f64"), 0L, 0L, 1L)
    nv_reduce_sum(y, dims = 1L, drop = TRUE)
  }))
  x <- nv_array(c(1, 2, 3), dtype = "f64")
  g <- f(x)
  expect_equal(g[[1L]], nv_array(1:3, dtype = "f64"))

  # Test with both edge and interior padding
  f2 <- jit(gradient(function(x) {
    x <- x * nv_array(c(1, 2), dtype = "f64")
    # edge_padding_low=1, edge_padding_high=1, interior_padding=1
    # For input [a, b], output is [0, a, 0, b, 0]
    y <- prim_pad(x, nv_scalar(0, "f64"), 1L, 1L, 1L)
    nv_reduce_sum(y, dims = 1L, drop = TRUE)
  }))
  x2 <- nv_array(c(5, 10), dtype = "f64")
  g2 <- f2(x2)
  expect_equal(g2[[1L]], nv_array(c(1, 2), dtype = "f64"))

  # Test 2D with interior padding
  f3 <- jit(gradient(function(x) {
    x <- x * nv_array(c(1, 2, 3, 4), shape = c(2, 2), dtype = "f64")
    y <- prim_pad(x, nv_scalar(0, "f64"), c(0L, 0L), c(0L, 0L), c(1L, 1L))
    nv_reduce_sum(y, dims = c(1L, 2L), drop = TRUE)
  }))
  x3 <- nv_matrix(1:4, nrow = 2, ncol = 2, dtype = "f64")
  g3 <- f3(x3)
  expect_equal(g3[[1L]], nv_matrix(1:4, nrow = 2, ncol = 2, dtype = "f64"))

  # Test 2D with different edge padding on each dimension
  f4 <- jit(gradient(function(x) {
    y <- prim_pad(x, nv_scalar(0, "f64"), c(1L, 2L), c(2L, 1L), c(0L, 0L))
    nv_reduce_sum(y, dims = c(1L, 2L), drop = TRUE)
  }))
  x4 <- nv_matrix(1:6, nrow = 2, ncol = 3, dtype = "f64")
  g4 <- f4(x4)
  expect_equal(g4[[1L]], nv_matrix(rep(1, 6), nrow = 2, ncol = 3, dtype = "f64"))
})

test_that("prim_dynamic_slice reverse", {
  # Gradient should scatter the incoming gradient back to the operand position
  f <- jit(gradient(
    function(x, start_i) {
      sliced <- prim_dynamic_slice(x, start_i, slice_sizes = 3L)
      nv_reduce_sum(sliced, dims = 1L, drop = TRUE)
    },
    wrt = "x"
  ))

  x <- nv_array(1:10, dtype = "f64", shape = 10L)
  start_i <- nv_scalar(3L, dtype = "i32")
  grad <- f(x, start_i)[[1L]]

  # Gradient is 1 at positions 3, 4, 5 (the sliced region) and 0 elsewhere
  expect_equal(grad, nv_array(c(0, 0, 1, 1, 1, 0, 0, 0, 0, 0), dtype = "f64", shape = 10L))

  # Test 2D case
  f2d <- jit(gradient(
    function(x, start_i, start_j) {
      sliced <- prim_dynamic_slice(x, start_i, start_j, slice_sizes = c(2L, 2L))
      nv_reduce_sum(sliced, dims = c(1L, 2L), drop = TRUE)
    },
    wrt = "x"
  ))

  x2d <- nv_array(1:12, dtype = "f64", shape = c(3L, 4L))
  start_i <- nv_scalar(2L, dtype = "i32")
  start_j <- nv_scalar(2L, dtype = "i32")
  grad2d <- f2d(x2d, start_i, start_j)[[1L]]

  # Gradient is 1 at positions (2,2), (2,3), (3,2), (3,3) and 0 elsewhere
  expect_equal(grad2d, nv_array(c(0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0), dtype = "f64", shape = c(3, 4)))
})

test_that("prim_dynamic_slice reverse with out-of-bounds", {
  # Test that gradient works correctly when indices are clamped
  f <- jit(gradient(
    function(x, start_i) {
      sliced <- prim_dynamic_slice(x, start_i, slice_sizes = c(5L))
      nv_reduce_sum(sliced, dims = 1L, drop = TRUE)
    },
    wrt = "x"
  ))

  x <- nv_array(1:10, dtype = "f64", shape = 10L)
  # Request slice at position 8 with size 5, will be clamped to position 6
  start_i <- nv_scalar(8L, dtype = "i32")
  grad <- f(x, start_i)[[1L]]

  # Gradient should be 1 at positions 6-10 (the actual sliced region after clamping)
  expect_equal(grad, nv_array(c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1), dtype = "f64", shape = 10L))
})

test_that("prim_dynamic_update_slice reverse", {
  # Test gradient for operand (zero out the updated region)
  f_operand <- jit(gradient(
    function(x, update, start_i) {
      updated <- prim_dynamic_update_slice(x, update, start_i)
      nv_reduce_sum(updated, dims = 1L, drop = TRUE)
    },
    wrt = "x"
  ))

  x <- nv_array(1:10, dtype = "f64", shape = 10L)
  update <- nv_array(c(100, 200, 300), dtype = "f64", shape = 3L)
  start_i <- nv_scalar(4L, dtype = "i32")
  grad_x <- f_operand(x, update, start_i)[[1L]]

  # Gradient is 0 at positions 4, 5, 6 (the updated region) and 1 elsewhere
  expect_equal(grad_x, nv_array(c(1, 1, 1, 0, 0, 0, 1, 1, 1, 1), dtype = "f64", shape = 10L))

  # Test gradient for update (slice out the corresponding region from grad)
  f_update <- jit(gradient(
    function(x, update, start_i) {
      updated <- prim_dynamic_update_slice(x, update, start_i)
      nv_reduce_sum(updated, dims = 1L, drop = TRUE)
    },
    wrt = "update"
  ))

  grad_update <- f_update(x, update, start_i)[[1L]]

  # Gradient for update is all 1s (since we sum everything)
  expect_equal(grad_update, nv_array(c(1, 1, 1), dtype = "f64", shape = 3L))
})

test_that("prim_dynamic_update_slice reverse with out-of-bounds", {
  # Test that gradient works correctly when indices are clamped
  f_operand <- jit(gradient(
    function(x, update, start_i) {
      updated <- prim_dynamic_update_slice(x, update, start_i)
      nv_reduce_sum(updated, dims = 1L, drop = TRUE)
    },
    wrt = "x"
  ))

  x <- nv_array(1:10, dtype = "f64", shape = 10L)
  update <- nv_array(c(100, 200, 300, 400, 500), dtype = "f64", shape = 5L)
  # Request update at position 8 with size 5, will be clamped to position 6
  start_i <- nv_scalar(8L, dtype = "i32")
  grad_x <- f_operand(x, update, start_i)[[1L]]

  # Gradient is 0 at positions 6-10 (the actual updated region after clamping) and 1 elsewhere
  expect_equal(grad_x, nv_array(c(1, 1, 1, 1, 1, 0, 0, 0, 0, 0), dtype = "f64", shape = 10L))

  # Test gradient for update with out-of-bounds
  f_update <- jit(gradient(
    function(x, update, start_i) {
      updated <- prim_dynamic_update_slice(x, update, start_i)
      nv_reduce_sum(updated, dims = 1L, drop = TRUE)
    },
    wrt = "update"
  ))

  grad_update <- f_update(x, update, start_i)[[1L]]

  # Gradient for update is all 1s (since we sum everything from the clamped position)
  expect_equal(grad_update, nv_array(c(1, 1, 1, 1, 1), dtype = "f64", shape = 5L))
})

test_that("prim_is_finite", {
  x <- nv_array(c(1.0, Inf, -Inf, NaN, 0.5, -0.5), dtype = "f32")
  verify_zero_grad_unary(prim_is_finite, x)
})

test_that("prim_popcnt", {
  x <- nv_array(c(0L, 1L, 3L, 7L, 15L), dtype = "i32")
  verify_zero_grad_unary(prim_popcnt, x)
})

describe("shift ops", {
  it("prim_shift_left returns zero gradients", {
    x <- nv_array(c(1L, 2L, 4L, 8L), dtype = "i32")
    y <- nv_array(c(1L, 1L, 1L, 1L), dtype = "i32")
    verify_zero_grad_binary(prim_shift_left, x, y)
  })

  it("prim_shift_right_arithmetic returns zero gradients", {
    x <- nv_array(c(8L, 16L, 32L, -8L), dtype = "i32")
    y <- nv_array(c(1L, 2L, 1L, 1L), dtype = "i32")
    verify_zero_grad_binary(prim_shift_right_arithmetic, x, y)
  })

  it("prim_shift_right_logical returns zero gradients", {
    x <- nv_array(c(8L, 16L, 32L, 64L), dtype = "i32")
    y <- nv_array(c(1L, 2L, 1L, 2L), dtype = "i32")
    verify_zero_grad_binary(prim_shift_right_logical, x, y)
  })
})

test_that("prim_bitcast_convert", {
  x <- nv_array(c(1.0, 2.0, 3.0, 4.0), dtype = "f32")
  verify_zero_grad_unary(prim_bitcast_convert, x, f_wrapper = function(x) {
    out <- prim_bitcast_convert(x, dtype = "i32")
    out <- nv_convert(out, "f32")
    nv_reduce_sum(out, dims = 1L, drop = TRUE)
  })
})

describe("boolean ops", {
  expected_zeros <- nv_array(rep(0, 4), dtype = "f32")

  verify_bool_binary <- function(prim_fn) {
    x <- nv_array(c(1, 0, 0, 1))
    y <- nv_array(c(1, 1, 0, 0))
    f <- function(x, y) {
      x <- nv_convert(x, "bool")
      y <- nv_convert(y, "bool")
      out <- prim_fn(x, y)
      nv_convert(out, "f32")
    }
    verify_zero_grad_binary(f, x, y)
  }

  verify_bool_reduce <- function(prim_fn) {
    x <- nv_array(c(1.0, 1.0, 0.0, 0.0), dtype = "f32")
    f <- function(x) {
      x_pred <- nv_convert(x, "bool")
      out <- prim_fn(x_pred, dims = 1L, drop = TRUE)
      nv_convert(out, "f32")
    }
    grads <- jit(gradient(f))(x)
    testthat::expect_equal(grads[[1L]], expected_zeros)
  }

  it("prim_and returns zero gradients", {
    verify_bool_binary(prim_and)
  })
  it("prim_or returns zero gradients", {
    verify_bool_binary(prim_or)
  })
  it("prim_xor returns zero gradients", {
    verify_bool_binary(prim_xor)
  })
  it("prim_not returns zero gradients", {
    x <- nv_array(c(1.0, 0.0, 1.0, 0.0), dtype = "f32")
    f <- function(x) {
      x_pred <- nv_convert(x, "bool")
      out <- prim_not(x_pred)
      out <- nv_convert(out, "f32")
      nv_reduce_sum(out, dims = 1L, drop = TRUE)
    }
    verify_zero_grad_unary(prim_not, x, f_wrapper = f)
  })
  it("prim_reduce_all returns zero gradients", {
    verify_bool_reduce(prim_reduce_all)
  })
  it("prim_reduce_any returns zero gradients", {
    verify_bool_reduce(prim_reduce_any)
  })
})

describe("prim_gather", {
  it("out of bounds", {
    out <- jit_eval({
      x <- nv_array(1:4, "f32")
      g1 <- gradient(function(x) {
        mean(x[nv_array(5:7)]^2)
      })(x)
      g2 <- gradient(function(x) {
        mean(x[array(c(4L, 4L, 4L, 4L))]^2)
      })(x)
      list(g1[[1L]], g2[[1L]])
    })
    expect_equal(out[[1]], out[[2]])
  })

  it("clamps out-of-range indices for gradient (matches forward clamping)", {
    # Index 10 on a size-4 array is clamped to 4 on forward pass.
    # The gradient should flow to the clamped position (4), not 10.
    f <- jit(gradient(function(x) {
      idx <- nv_scalar(10L, dtype = "i32")
      nv_subset(x, idx)
    }))
    x <- nv_array(c(1, 2, 3, 4), dtype = "f64")
    grads <- f(x)
    expect_equal(grads[[1L]], nv_array(c(0, 0, 0, 1), dtype = "f64"))
  })
})

describe("prim_scatter", {
  it("non-unique indices: only winning update gets gradient", {
    update <- nv_array(1:10, dtype = "f32")
    f <- function(update) {
      x <- nv_array(0)
      x[array(rep(1L, 10))] <- update
      mean(x^2)
    }
    g <- jit(\(update) {
      out <- value_and_gradient(f)(update)
      out[[1L]] <- sqrt(out[[1L]]) * 2
      out[[2]][[1]] <- sum(out[[2]][[1]]) # just get rid of the zeros
      out
    })(update)
    expect_equal(g[[1]], g[[2]][[1]])
  })

  it("errors for non-simple replacement update_computation", {
    expect_error(
      jit(gradient(function(x) {
        out <- prim_scatter(
          input = x,
          scatter_indices = nv_array(2L, dtype = "i64"),
          update = nv_scalar(10, dtype = "f32"),
          update_window_dims = integer(),
          inserted_window_dims = 1L,
          input_batching_dims = integer(),
          scatter_indices_batching_dims = integer(),
          scatter_dims_to_operand_dims = 1L,
          index_vector_dim = 1L,
          indices_are_sorted = TRUE,
          unique_indices = TRUE,
          update_computation = function(old, new) prim_add(old, new)
        )
        nv_reduce_sum(out, dims = 1L, drop = TRUE)
      }))(nv_array(1:5, dtype = "f32")),
      "simple replacement"
    )
  })
})

describe("gather/scatter reverse via subset operators", {
  check <- function(shape, ...) {
    quos <- rlang::enquos(...)

    spec <- parse_subset_specs(quos, shape)
    value_shape <- subset_spec_to_shape(spec)

    x <- nv_array(rnorm(prod(shape)), dtype = "f32", shape = shape)
    value <- nv_array(rnorm(prod(value_shape)), dtype = "f32", shape = value_shape)

    # Check gather
    f1 <- function(x, value) {
      out <- gradient(function(x, value) {
        y <- rlang::inject(nv_subset(x, !!!quos))
        mean(y * value)
      })(x, value)
    }

    f2 <- function(x_subset, value) {
      x_subset <- rlang::inject(nv_subset(x, !!!quos))
      out <- gradient(\(x, value) {
        mean(x_subset * value)
      })(x_subset, value)
      g1 <- nv_fill(0, shape = shape)
      out[[1L]] <- rlang::inject(nv_subset_assign(g1, !!!quos, value = out[[1]]))
      out
    }

    expect_equal(
      jit(f1)(x, value),
      jit(f2)(x, value)
    )

    # Check scatter
    f3 <- function(x, value) {
      gradient(function(x, value) {
        y <- nv_subset_assign(x, !!!quos, value = value)
        sum(y^2)
      })(x, value)
    }

    f4 <- function(x, value) {
      x2 <- rlang::inject(nv_subset_assign(x, !!!quos, value = 0))
      gradient(\(x, value) {
        sum(x^2) + sum(value^2)
      })(x2, value)
    }

    expect_equal(
      jit(f3)(x, value),
      jit(f4)(x, value)
    )
  }

  it("1D: single element", {
    check(c(8L), 3L)
  })

  it("1D: range", {
    check(c(10L), 2:5)
  })

  it("1D: full", {
    check(c(6L), )
  })

  it("1D: gather", {
    check(c(10L), array(c(1L, 4L, 7L)))
  })

  it("2D: single in both dims (scalar gather)", {
    check(c(4L, 5L), 2L, 3L)
  })

  it("2D: range + full", {
    check(c(6L, 4L), 2:4, )
  })

  it("2D: single in first, range in second", {
    check(c(5L, 8L), 3L, 2:6)
  })

  it("2D: gather in first, full second", {
    check(c(6L, 4L), array(c(1L, 3L, 5L)), )
  })

  it("2D: gather in both dims", {
    check(c(5L, 6L), array(c(1L, 3L)), array(c(2L, 4L)))
  })

  it("3D: range, single, full", {
    check(c(4L, 5L, 3L), 1:3, 2L, )
  })
})

# argmax / argmin reverse rule short-circuits the gradient to zero.
test_that("prim_argmax / prim_argmin have zero gradient", {
  x <- nv_array(c(3, 1, 4), dtype = "f32")
  for (prim in list(prim_argmax, prim_argmin)) {
    verify_zero_grad_unary(prim, x, f_wrapper = function(x) {
      out <- prim(x, dim = 1L)
      prim_convert(out, "f32", ambiguous = FALSE)
    })
  }
})

# prim_cummax / prim_cummin: tested manually because the R torch package's
# `torch_cummax` / `torch_cummin` wrappers call into a missing lantern symbol
# (`cpp_torch_namespace_cummax_self_Tensor`), so a torch comparison is not
# available.
describe("prim_cummax", {
  verify_grad <- function(x_arr, dim, expected_grad) {
    x_nv <- nv_array(x_arr, dtype = "f32")
    f_nv <- function(x) {
      y <- nv_cummax(x, dim = dim)
      nv_reduce_sum(y, dims = seq_along(shape(y)), drop = TRUE)
    }
    grads_nv <- jit(gradient(f_nv))(x_nv)
    expected <- if (length(dim(x_arr))) {
      array(expected_grad, dim = dim(x_arr))
    } else {
      array(expected_grad, dim = length(x_arr))
    }
    expect_equal(as_array(grads_nv[[1L]]), expected, tolerance = 1e-4)
  }
  it("strictly increasing -> each element receives its own grad", {
    verify_grad(c(1, 2, 3, 4), dim = 1L, c(1, 1, 1, 1))
  })
  it("plateau uses last occurrence", {
    # x = [1, 3, 3, 2]: y = [1, 3, 3, 3]. arg_run = [1, 2, 3, 3]. grad = [1, 1, 2, 0].
    verify_grad(c(1, 3, 3, 2), dim = 1L, c(1, 1, 2, 0))
  })
  it("non-monotone vector", {
    # x = [3, 1, 4, 1, 5, 9, 2, 6]: arg_run = [1, 1, 3, 3, 5, 6, 6, 6] (no ties).
    # grad = [2, 0, 2, 0, 1, 3, 0, 0].
    verify_grad(c(3, 1, 4, 1, 5, 9, 2, 6), dim = 1L, c(2, 0, 2, 0, 1, 3, 0, 0))
  })
  it("matrix gradient along dim 2", {
    # Matrix:  row 1 = c(1, 3, 2)  -> arg_run [1, 2, 2] -> grad [1, 2, 0]
    #          row 2 = c(4, 2, 1)  -> arg_run [1, 1, 1] -> grad [3, 0, 0]
    verify_grad(
      matrix(c(1, 4, 3, 2, 2, 1), nrow = 2),
      dim = 2L,
      matrix(c(1, 3, 2, 0, 0, 0), nrow = 2)
    )
  })
})

describe("prim_cummin", {
  verify_grad <- function(x_arr, dim, expected_grad) {
    x_nv <- nv_array(x_arr, dtype = "f32")
    f_nv <- function(x) {
      y <- nv_cummin(x, dim = dim)
      nv_reduce_sum(y, dims = seq_along(shape(y)), drop = TRUE)
    }
    grads_nv <- jit(gradient(f_nv))(x_nv)
    expected <- if (length(dim(x_arr))) {
      array(expected_grad, dim = dim(x_arr))
    } else {
      array(expected_grad, dim = length(x_arr))
    }
    expect_equal(as_array(grads_nv[[1L]]), expected, tolerance = 1e-4)
  }
  it("strictly decreasing -> each element receives its own grad", {
    verify_grad(c(4, 3, 2, 1), dim = 1L, c(1, 1, 1, 1))
  })
  it("plateau uses last occurrence", {
    # x = [4, 2, 2, 3]: y = [4, 2, 2, 2]. arg_run = [1, 2, 3, 3]. grad = [1, 1, 2, 0].
    verify_grad(c(4, 2, 2, 3), dim = 1L, c(1, 1, 2, 0))
  })
  it("non-monotone vector", {
    # x = [3, 1, 4, 1, 5, 9, 2, 6]: arg_run = [1, 2, 2, 4, 4, 4, 4, 4]
    # (tie at j=4 picks the later index, then carries forward).
    # grad = [1, 2, 0, 5, 0, 0, 0, 0].
    verify_grad(c(3, 1, 4, 1, 5, 9, 2, 6), dim = 1L, c(1, 2, 0, 5, 0, 0, 0, 0))
  })
  it("matrix gradient along dim 1", {
    # Matrix:  col 1 = c(3, 1, 4) -> arg_run [1, 2, 2] -> grad [1, 2, 0]
    #          col 2 = c(1, 5, 9) -> arg_run [1, 1, 1] -> grad [3, 0, 0]
    verify_grad(
      matrix(c(3, 1, 4, 1, 5, 9), nrow = 3),
      dim = 1L,
      matrix(c(1, 2, 0, 3, 0, 0), nrow = 3)
    )
  })
})

# prim_sort / prim_top_k: gradient compared to a manual scatter reference.
# torch's autograd raised a spurious "saved tensor modified inplace" version
# mismatch on the indices tensor returned by torch_sort/torch_topk, so we do
# not compare against torch here.
test_that("prim_sort", {
  withr::local_seed(42)
  # 2D descending sort along dim != 1. Distinct values so the result is
  # deterministic under the default is_stable = FALSE (tie-breaking unspecified).
  x_arr <- matrix(rnorm(4 * 6), nrow = 4)
  w_arr <- matrix(as.double(seq_len(4 * 6)), nrow = 4)

  x_nv <- nv_array(x_arr)
  w_nv <- nv_array(w_arr)

  f_nv <- function(x) {
    sorted <- prim_sort(list(x), dim = 2L, descending = TRUE)[[1L]]
    nv_reduce_sum(sorted * w_nv, dims = c(1L, 2L))
  }
  grad_nv <- jit(gradient(f_nv))(x_nv)[[1L]]

  # For each row, scatter w back along the descending sort indices:
  # grad[i, sorted_idx[k]] = w[i, k].
  expected_grad <- matrix(0, nrow = nrow(x_arr), ncol = ncol(x_arr))
  for (i in seq_len(nrow(x_arr))) {
    sorted_idx <- order(x_arr[i, ], decreasing = TRUE)
    expected_grad[i, sorted_idx] <- w_arr[i, ]
  }

  expect_equal(as_array(grad_nv), expected_grad, tolerance = 1e-5)
})

test_that("prim_top_k", {
  withr::local_seed(42)
  x_arr <- matrix(rnorm(4 * 6), nrow = 4)
  k <- 3L
  w_arr <- matrix(as.double(seq_len(4 * k)), nrow = 4)

  x_nv <- nv_array(x_arr)
  w_nv <- nv_array(w_arr)

  f_nv <- function(x) {
    top <- prim_top_k(x, k = k)[[1L]]
    nv_reduce_sum(top * w_nv, dims = c(1L, 2L))
  }
  grad_nv <- jit(gradient(f_nv))(x_nv)[[1L]]

  # Scatter w along the top-k indices for each row.
  expected_grad <- matrix(0, nrow = nrow(x_arr), ncol = ncol(x_arr))
  for (i in seq_len(nrow(x_arr))) {
    top_idx <- order(x_arr[i, ], decreasing = TRUE)[seq_len(k)]
    expected_grad[i, top_idx] <- w_arr[i, ]
  }

  expect_equal(as_array(grad_nv), expected_grad, tolerance = 1e-5)
})

test_that("prim_reduce_prod: gradient is safe at zeros", {
  # The safe rule: gradient at position i = product of all *other* elements
  # along the reduced axis. With one zero, only the zero position has a
  # non-zero gradient; with two or more zeros, the gradient is zero everywhere.
  f1 <- jit(gradient(function(x) prim_reduce_prod(x, dims = 1L, drop = TRUE)))

  expect_equal(as.numeric(f1(nv_array(c(2, 3, 5)))[[1L]]), c(15, 10, 6))
  expect_equal(as.numeric(f1(nv_array(c(2, 0, 5)))[[1L]]), c(0, 10, 0))
  expect_equal(as.numeric(f1(nv_array(c(2, 0, 0)))[[1L]]), c(0, 0, 0))
})

test_that("prim_reduce_prod: multi-axis reduction with a zero", {
  x <- matrix(c(2, 3, 0, 4, 5, 6), nrow = 3, byrow = TRUE)
  f <- jit(gradient(function(x) prim_reduce_prod(x, dims = c(1L, 2L), drop = TRUE)))
  expected <- matrix(0, nrow = 3, ncol = 2)
  expected[2, 1] <- prod(x[x != 0])
  expect_equal(as_array(f(nv_array(x))[[1L]]), expected)
})

test_that("prim_reduce_prod: drop = FALSE matches drop = TRUE", {
  x <- nv_matrix(c(2, 3, 0, 4, 5, 6), nrow = 3, byrow = TRUE)
  f_drop <- jit(gradient(function(x) {
    nv_reduce_sum(prim_reduce_prod(x, dims = 2L, drop = TRUE), dims = 1L, drop = TRUE)
  }))
  f_keep <- jit(gradient(function(x) {
    nv_reduce_sum(prim_reduce_prod(x, dims = 2L, drop = FALSE), dims = c(1L, 2L), drop = TRUE)
  }))
  expect_equal(as_array(f_drop(x)[[1L]]), as_array(f_keep(x)[[1L]]))
})

if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "test-primitives-reverse-torch.R", package = "anvl"), local = TRUE)
}
