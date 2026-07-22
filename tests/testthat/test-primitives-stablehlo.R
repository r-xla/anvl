test_that("prim_sin", {
  x <- nv_array(c(0, pi / 2, pi, 3 / 2 * pi), dtype = "f64")
  out <- as_array(prim_sin(x))
  expect_equal(c(out), c(0, 1, 0, -1), tolerance = 1e-15)
})

test_that("prim_cos", {
  x <- nv_array(c(0, pi / 2, pi, 3 / 2 * pi), dtype = "f64")
  out <- as_array(prim_cos(x))
  expect_equal(c(out), c(1, 0, -1, 0), tolerance = 1e-15)
})

test_that("prim_rng_bit_generator", {
  out <- prim_rng_bit_generator(nv_array(c(1, 2), dtype = "ui64"), "THREE_FRY", "i64", c(2, 2))
  expect_equal(dtype(out[[1]]), as_dtype("ui64"))
  expect_equal(shape(out[[1]]), 2L)
  expect_equal(shape(out[[2]]), c(2L, 2L))
})

test_that("prim_bitcast_convert", {
  out <- nv_bitcast_convert(
    nv_array(seq(-1, 1, length.out = 6), dtype = "f64", shape = c(2, 3)),
    dtype = "i32"
  )
  expect_equal(dim(as_array(out)), c(2, 3, 2))
  expect_true(is.integer(as_array(out)))
})

test_that("prim_static_slice", {
  out <- nv_static_slice(
    nv_array(1:6, dtype = "ui64", shape = c(2, 3)),
    start_indices = c(1, 1),
    limit_indices = c(2, 2),
    strides = c(1, 1)
  )
  expect_equal(as_array(out), matrix(c(1:4), nrow = 2))
})

test_that("prim_dynamic_slice", {
  # Basic dynamic slice with scalar indices
  x <- nv_array(1:12, dtype = "i32", shape = c(3, 4))
  # Slice starting at (1, 1) should give [[1, 4], [2, 5]]
  out <- prim_dynamic_slice(x, nv_scalar(1L, dtype = "i32"), nv_scalar(1L, dtype = "i32"), slice_sizes = c(2L, 2L))
  expect_equal(out, nv_array(c(1L, 2L, 4L, 5L), dtype = "i32", shape = c(2, 2)))

  # Slice starting at (2, 2) should give [[5, 8], [6, 9]]
  out <- prim_dynamic_slice(x, nv_scalar(2L, dtype = "i32"), nv_scalar(2L, dtype = "i32"), slice_sizes = c(2L, 2L))
  expect_equal(out, nv_array(c(5L, 6L, 8L, 9L), dtype = "i32", shape = c(2, 2)))

  # 1D case
  x1d <- nv_array(1:10, dtype = "i32", shape = c(10))
  out <- prim_dynamic_slice(x1d, nv_scalar(3L, dtype = "i32"), slice_sizes = c(3L))
  expect_equal(out, nv_array(c(3L, 4L, 5L), dtype = "i32", shape = 3L))
})

test_that("prim_dynamic_update_slice", {
  expect_equal(
    prim_dynamic_update_slice(nv_scalar(1L, dtype = "i32"), nv_scalar(100L, dtype = "i32")),
    nv_scalar(100L, dtype = "i32")
  )

  # Basic dynamic update slice with scalar indices
  x <- nv_array(1:12, dtype = "i32", shape = c(3, 4))
  update <- nv_array(c(100L, 200L, 300L, 400L), dtype = "i32", shape = c(2, 2))

  # Update at (1, 1) - top-left corner
  out <- prim_dynamic_update_slice(x, update, nv_scalar(1L, dtype = "i32"), nv_scalar(1L, dtype = "i32"))
  expect_equal(
    out,
    nv_array(c(100L, 200L, 3L, 300L, 400L, 6L, 7L, 8L, 9L, 10L, 11L, 12L), dtype = "i32", shape = c(3, 4))
  )

  # Update at (2, 3) - bottom-right corner
  out <- prim_dynamic_update_slice(x, update, nv_scalar(2L, dtype = "i32"), nv_scalar(3L, dtype = "i32"))
  expect_equal(
    out,
    nv_array(c(1L, 2L, 3L, 4L, 5L, 6L, 7L, 100L, 200L, 10L, 300L, 400L), dtype = "i32", shape = c(3, 4))
  )

  # 1D case
  x1d <- nv_array(1:10, dtype = "i32", shape = c(10))
  update1d <- nv_array(c(100L, 200L, 300L), dtype = "i32", shape = c(3))
  out <- prim_dynamic_update_slice(x1d, update1d, nv_scalar(4L, dtype = "i32"))
  expect_equal(out, nv_array(c(1L, 2L, 3L, 100L, 200L, 300L, 7L, 8L, 9L, 10L), dtype = "i32", shape = 10L))
})

test_that("prim_concatenate", {
  out <- nv_concatenate(
    nv_array(c(1:6), dtype = "ui64", shape = c(2, 3)),
    nv_array(c(7:10), dtype = "ui64", shape = c(2, 2)),
    dimension = 2L
  )
  expect_equal(dim(as_array(out)), c(2, 5))
})
test_that("prim_fill", {
  f <- jit(function(x) nv_fill(x, shape = c(2, 3), dtype = "f32"), static = "x")
  expect_equal(f(1), nv_array(1, shape = c(2, 3), dtype = "f32"))
  expect_equal(f(2), nv_array(2, shape = c(2, 3), dtype = "f32"))

  # scalars
  expect_equal(
    nv_fill(1L, shape = c(), dtype = "f32"),
    nv_scalar(1, dtype = "f32")
  )
  expect_equal(
    nv_fill(1L, shape = integer(), dtype = "f32"),
    nv_scalar(1, dtype = "f32")
  )
  expect_equal(
    nv_fill(1L, shape = 1L, dtype = "f32"),
    nv_array(1, shape = 1L, dtype = "f32")
  )
})

test_that("prim_shift_left", {
  x <- nv_array(as.integer(c(1L, 2L, 3L, 8L)), dtype = "i32")
  y <- nv_array(as.integer(c(0L, 1L, 2L, 3L)), dtype = "i32")
  out <- as.integer(prim_shift_left(x, y))
  expect_equal(out, as.integer(c(1L, 4L, 12L, 64L)))
})

test_that("prim_shift_right_logical", {
  x <- nv_array(as.integer(c(16L, 8L, 7L, 1L)), dtype = "i32")
  y <- nv_array(as.integer(c(0L, 1L, 2L, 0L)), dtype = "i32")
  out <- as.integer(prim_shift_right_logical(x, y))
  expect_equal(out, as.integer(c(16L, 4L, 1L, 1L)))
})

test_that("prim_shift_right_arithmetic", {
  x <- nv_array(as.integer(c(-8L, -1L, 8L, -17L)), dtype = "i32")
  y <- nv_array(as.integer(c(1L, 3L, 2L, 4L)), dtype = "i32")
  out <- as.integer(prim_shift_right_arithmetic(x, y))
  expect_equal(out, as.integer(c(-4L, -1L, 2L, -2L)))
})

# Reduction ops (simplified hardcoded examples, no torch comparisons)

test_that("prim_reduce_sum", {
  x <- array(1:6, c(2, 3))
  out <- as_array(prim_reduce_sum(nv_array(x, dtype = "f32"), dims = 2L, drop = TRUE))
  expect_equal(out, array(c(9, 12)))
})

test_that("prim_reduce_prod", {
  x <- array(1:6, c(2, 3))
  out <- as_array(prim_reduce_prod(nv_array(x, dtype = "f32"), dims = 1L, drop = FALSE))
  expect_equal(out, array(c(2, 12, 30), c(1, 3)))
})

test_that("prim_reduce_max", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  out <- as_array(prim_reduce_max(nv_array(x, dtype = "f32"), dims = 2L, drop = TRUE))
  expect_equal(out, array(c(0, 4)))
  # f64
  x <- nv_reduce_max(nv_array(c(1, 2, 3), dtype = "f64"), dims = 1L)
  expect_equal(x, nv_scalar(3, dtype = "f64"))
})

test_that("prim_reduce_max drop = FALSE", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  out <- as_array(prim_reduce_max(nv_array(x, dtype = "f32"), dims = 2L, drop = FALSE))
  expect_equal(out, array(c(0, 4), c(2, 1)))
})

test_that("prim_reduce_min", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  out <- as_array(prim_reduce_min(nv_array(x, dtype = "f32"), dims = 2L, drop = TRUE))
  expect_equal(out, array(c(-1, 2)))
  # f64
  x <- nv_reduce_min(nv_array(c(1, 2, 3), dtype = "f64"), dims = 1L)
  expect_equal(x, nv_scalar(1, dtype = "f64"))
})

test_that("prim_reduce_min drop = FALSE", {
  x <- array(c(-1, 4, 0, 2), c(2, 2))
  out <- as_array(prim_reduce_min(nv_array(x, dtype = "f32"), dims = 2L, drop = FALSE))
  expect_equal(out, array(c(-1, 2), c(2, 1)))
})

test_that("reductions over a zero-size axis return the identity", {
  empty1 <- nv_array(numeric(0), shape = 0L, dtype = "f32")
  empty1_bool <- nv_array(logical(0), shape = 0L, dtype = "bool")
  expect_equal(as_array(prim_reduce_sum(empty1, dims = 1L, drop = TRUE)), 0)
  expect_equal(as_array(prim_reduce_prod(empty1, dims = 1L, drop = TRUE)), 1)
  expect_equal(as_array(prim_reduce_any(empty1_bool, dims = 1L, drop = TRUE)), FALSE)
  expect_equal(as_array(prim_reduce_all(empty1_bool, dims = 1L, drop = TRUE)), TRUE)

  # Reducing along an empty axis of a higher-rank tensor keeps the other axes.
  empty2 <- nv_array(numeric(0), shape = c(2L, 0L), dtype = "f32")
  expect_equal(as_array(prim_reduce_sum(empty2, dims = 2L, drop = TRUE)), array(c(0, 0), 2L))
  # Reducing a non-empty axis of a tensor with a separate empty axis is fine too.
  out <- as_array(prim_reduce_max(empty2, dims = 1L, drop = TRUE))
  expect_equal(dim(out), 0L)
})

test_that("prim_reduce_any", {
  x <- array(c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE), c(2, 3))
  out <- as_array(prim_reduce_any(nv_array(x, dtype = "bool"), dims = 2L, drop = TRUE))
  expect_equal(out, array(c(TRUE, FALSE)))
})

test_that("prim_reduce_all", {
  x <- array(c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE), c(2, 3))
  out <- as_array(prim_reduce_all(nv_array(x, dtype = "bool"), dims = 1L, drop = FALSE))
  expect_equal(out, array(rep(FALSE, 3), c(1, 3)))
})

describe("cumulative ops", {
  # Common semantics across cumsum / cumprod / cummax / cummin.
  # cummax / cummin return list(values, indices); the helper picks values.
  verify_cum <- function(prim_fn, nv_fn, base_fn, has_indices = FALSE) {
    pick <- if (has_indices) function(o) o[[1L]] else function(o) o

    # 1-D: matches base R directly.
    v <- c(3, 1, 4, 1, 5, 9, 2, 6)
    x <- nv_array(v, dtype = "f32")
    expect_equal(as_array(pick(prim_fn(x, dim = 1L))), array(base_fn(v)))

    # 2-D dim = 1 (down columns) and dim = 2 (across rows) match
    # `apply` along the corresponding margin.
    M <- matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
    xm <- nv_array(M, dtype = "f32")

    expect_equal(as_array(pick(prim_fn(xm, dim = 1L))), apply(M, 2, base_fn))

    expect_equal(as_array(pick(prim_fn(xm, dim = 2L))), t(apply(M, 1, base_fn)))

    # row vs column major ordering
    expect_equal(as_array(nv_fn(xm)), array(base_fn(t(M))))
  }

  it("prim_cumsum matches base R", verify_cum(prim_cumsum, nv_cumsum, base::cumsum))
  it("prim_cumprod matches base R", verify_cum(prim_cumprod, nv_cumprod, base::cumprod))
  it("prim_cummax matches base R", verify_cum(prim_cummax, nv_cummax, base::cummax, has_indices = TRUE))
  it("prim_cummin matches base R", verify_cum(prim_cummin, nv_cummin, base::cummin, has_indices = TRUE))

  it("prim_cumsum rejects out-of-range dim", {
    x <- nv_array(1:4, dtype = "f32")
    expect_error(prim_cumsum(x, dim = 2L), "dim")
    expect_error(prim_cumsum(x, dim = 0L), "dim")
  })

  # Index outputs are unique to cummax / cummin -- not covered by the
  # values-only helper. Plateaus break ties to the last occurrence
  # (matching torch).
  it("prim_cummax returns running argmax indices", {
    x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6), dtype = "f32")
    out <- prim_cummax(x, dim = 1L)
    expect_equal(c(as_array(out[[2L]])), c(1L, 1L, 3L, 3L, 5L, 6L, 6L, 6L))
  })
  it("prim_cummin returns running argmin indices with last-occurrence tiebreak", {
    # Tie at j=4 (x_4 == y_3 == 1): last-occurrence picks 4, then carries forward.
    x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6), dtype = "f32")
    out <- prim_cummin(x, dim = 1L)
    expect_equal(c(as_array(out[[2L]])), c(1L, 2L, 2L, 4L, 4L, 4L, 4L, 4L))
  })
  it("prim_cummax plateau breaks ties to last occurrence", {
    x <- nv_array(c(1, 3, 3, 2), dtype = "f32")
    out <- prim_cummax(x, dim = 1L)
    expect_equal(c(as_array(out[[2L]])), c(1L, 2L, 3L, 3L))
  })
  it("prim_cummax integer dtype", {
    x <- nv_array(c(3L, -1L, 4L, -1L, 5L), dtype = "i32")
    out <- prim_cummax(x, dim = 1L)
    expect_equal(c(as_array(out[[1L]])), c(3L, 3L, 4L, 4L, 5L))
    expect_equal(c(as_array(out[[2L]])), c(1L, 1L, 3L, 3L, 5L))
  })
})

test_that("prim_broadcast_in_dim", {
  x <- 1L
  f <- jit(prim_broadcast_in_dim, static = c("shape", "broadcast_dimensions"))
  expect_equal(
    f(nv_scalar(1L), c(1, 2), integer()),
    nv_array(1L, shape = c(1, 2)),
    tolerance = 1e-5
  )
})

test_that("prim_reshape", {
  f <- jit(prim_reshape, static = "shape")
  x <- array(1:6, c(3, 2))
  expect_equal(
    f(nv_array(x), shape = 6),
    nv_array(as.integer(c(1, 4, 2, 5, 3, 6)), "i32")
  )
})

test_that("prim_transpose", {
  x <- array(1:4, c(2, 2))
  expect_equal(
    t(x),
    as_array(prim_transpose(nv_array(x), c(2, 1)))
  )
})

describe("prim_if", {
  it("can capture non-arguments", {
    f <- jit(function(pred, x) {
      x1 <- nv_mul(x, x)
      x2 <- nv_add(x, x)
      nv_if(pred, \() x1, \() x2)
    })
    expect_equal(
      f(nv_scalar(TRUE), nv_scalar(2)),
      nv_scalar(4)
    )
    expect_equal(
      f(nv_scalar(FALSE), nv_scalar(2)),
      nv_scalar(4)
    )
  })

  it("works in simple example", {
    # simple
    f <- function(pred, x) prim_if(pred, \() x, \() x * x)
    fj <- jit(f)
    expect_equal(fj(nv_scalar(TRUE), nv_scalar(2)), nv_scalar(2))
    expect_equal(fj(nv_scalar(FALSE), nv_scalar(2)), nv_scalar(4))
    graph <- trace_fn(f, list(pred = nv_scalar(TRUE), x = nv_scalar(2)))

    graph <- trace_fn(f, list(pred = nv_scalar(TRUE), x = nv_scalar(2)))

    f <- jit(function(pred, x) {
      prim_if(pred, \() list(list(x)), \() list(list(x * x)))
    })
    expect_equal(
      f(nv_scalar(TRUE), nv_scalar(2)),
      list(list(nv_scalar(2)))
    )
    expect_equal(
      f(nv_scalar(FALSE), nv_scalar(2)),
      list(list(nv_scalar(4)))
    )

    g <- jit(function(pred, x) {
      prim_if(pred, \() list(x[[1]]), \() list(x[[1]] * x[[1]]))
    })
    expect_equal(
      g(nv_scalar(FALSE), list(nv_scalar(2))),
      list(nv_scalar(4))
    )
  })

  it("identical constants in both branches receive the same GraphValue", {
    x <- nv_scalar(1)
    f <- function(y) prim_if(y, \() x, \() x)
    graph <- trace_fn(f, list(y = nv_scalar(TRUE)))
    fj <- jit(f)
    expect_equal(fj(nv_scalar(TRUE)), nv_scalar(1))
    expect_equal(fj(nv_scalar(FALSE)), nv_scalar(1))

    g <- jit(function(pred) {
      y <- nv_scalar(2)
      prim_if(pred, \() y, \() y * nv_scalar(3))
    })
    expect_equal(g(nv_scalar(TRUE)), nv_scalar(2))
    expect_equal(g(nv_scalar(FALSE)), nv_scalar(6))
  })

  it("works with literals as predicate", {
    expect_equal(nv_if(TRUE, \() 1, \() 2), nv_scalar(1, ambiguous = TRUE))
  })

  it("works with multi-element R array branch outputs", {
    f <- jit(function(pred) {
      nv_if(pred, \() array(c(1L, 2L, 3L)), \() array(c(4L, 5L, 6L)))
    })
    expect_equal(f(nv_scalar(TRUE)), nv_array(c(1L, 2L, 3L), ambiguous = TRUE))
    expect_equal(f(nv_scalar(FALSE)), nv_array(c(4L, 5L, 6L), ambiguous = TRUE))
  })
})


# TODO: Continue here
describe("prim_while", {
  it("works in simple case", {
    f <- jit(function(n) {
      nv_while(list(i = nv_scalar(1L)), \(i) i <= n, \(i) {
        i <- i + nv_scalar(1L)
        list(i = i)
      })
    })

    expect_equal(
      f(nv_scalar(10L)),
      list(i = nv_scalar(11L))
    )
  })

  it("can use literals in the loop", {
    f <- jit(function(n) {
      nv_while(list(i = nv_scalar(1L)), \(i) i <= n, \(i) {
        i <- i + 1L
        list(i = i)
      })
    })
    expect_equal(f(nv_scalar(10L)), list(i = nv_scalar(11L)))
  })

  it("works with two state variables", {
    f <- jit(function(n) {
      nv_while(
        list(i = nv_scalar(1L), s = nv_scalar(0L)),
        \(i, s) i <= n,
        \(i, s) {
          i <- i + nv_scalar(1L)
          s <- s + i
          list(i = i, s = s)
        }
      )
    })

    res <- f(nv_scalar(10L))
    expect_equal(
      res$i,
      nv_scalar(11L)
    )
    expect_equal(
      res$s,
      nv_scalar(sum(2:11))
    )
  })

  it("works with two states where one is unused", {
    f <- jit(function(n) {
      nv_while(
        list(i = nv_scalar(1L), j = nv_scalar(2L)),
        \(i, j) {
          # nolint
          i <= n
        },
        \(i, j) {
          i <- i + nv_scalar(1L)
          list(i = i, j = j)
        }
      ) # nolint
    })

    expect_equal(
      f(nv_scalar(10L)),
      list(i = nv_scalar(11L), j = nv_scalar(2L))
    )
  })

  it("works with nested state", {
    f <- jit(function(n) {
      nv_while(
        list(i = list(nv_scalar(1L))),
        \(i) {
          i[[1]] <= n
        },
        \(i) {
          i <- i[[1L]]
          i <- i + nv_scalar(1L)
          list(i = list(i))
        }
      )
    })
    expect_equal(
      f(nv_scalar(10L)),
      list(i = list(nv_scalar(11L)))
    )
  })

  it("works with literal initial state", {
    f <- jit(function(n, x) {
      i <- 1L
      out <- nv_while(list(i = i), \(i) i <= n, \(i) {
        i <- i + 1L
        list(i = i)
      })
      x + out$i
    })
    expect_equal(f(nv_scalar(10L), nv_scalar(5L)), nv_scalar(16L))
  })

  it("works with multi-element R array initial state", {
    f <- jit(function(n) {
      v <- array(c(1L, 2L, 3L))
      nv_while(list(i = nv_scalar(0L), v = v), \(i, v) i < n, \(i, v) {
        list(i = i + 1L, v = v + 1L)
      })
    })
    out <- f(nv_scalar(5L))
    expect_equal(out$i, nv_scalar(5L))
    expect_equal(out$v, nv_array(c(6L, 7L, 8L), ambiguous = TRUE))
  })

  it("errors", {
    # TODO:
  })
})

test_that("prim_chol", {
  A <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f64")
  L <- as_array(prim_chol(A, lower = TRUE))
  expect_equal(L[1, 1], 2)
  expect_equal(L[2, 1], 1)
  expect_equal(L[2, 2], sqrt(2), tolerance = 1e-5)
  # Verify L %*% t(L) = A
  expect_equal(L %*% t(L), matrix(c(4, 2, 2, 3), nrow = 2), tolerance = 1e-5)
})

test_that("prim_chol zeros out non-triangular part", {
  A <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f64")
  L <- as_array(prim_chol(A, lower = TRUE))
  expect_equal(L[1, 2], 0)

  U <- as_array(prim_chol(A, lower = FALSE))
  expect_equal(U[2, 1], 0)
})

test_that("prim_triangular_solve", {
  # Solve L %*% x = b where L = [[3, 0], [1, 2]]
  L <- nv_matrix(c(3, 1, 0, 2), nrow = 2, dtype = "f64")
  b <- nv_matrix(c(6, 5), nrow = 2, dtype = "f64")
  x <- as_array(prim_triangular_solve(
    L,
    b,
    left_side = TRUE,
    lower = TRUE,
    unit_diagonal = FALSE,
    transpose_a = FALSE
  ))
  # x = L^{-1} b: 3*x1 = 6 -> x1 = 2; x1 + 2*x2 = 5 -> x2 = 1.5
  expect_equal(c(x), c(2, 1.5), tolerance = 1e-5)

  # Verify: solve with transpose_a = TRUE
  x2 <- as_array(prim_triangular_solve(
    L,
    b,
    left_side = TRUE,
    lower = TRUE,
    unit_diagonal = FALSE,
    transpose_a = TRUE
  ))
  # L^T x = b: [[3,1],[0,2]] x = [6,5] -> 2*x2=5 -> x2=2.5; 3*x1+x2=6 -> x1=7/6
  expect_equal(c(x2), c(7 / 6, 2.5), tolerance = 1e-5)
})

# Sanity smoke tests for the linalg primitives.
# All of these custom calls are already properly tested in pjrt.
# Here we just test the anvl wiring.

describe("prim_qr", {
  it("decomposes a tall matrix", {
    # Use a rectangular (m > n) matrix so the reduced-QR shapes (m, k) and
    # (k, n) with k = min(m, n) are visibly different and worth checking.
    m <- 4L
    n <- 2L
    k <- min(m, n)
    A <- nv_matrix(c(12, 6, -4, 1, -51, 167, 24, 5), nrow = m, dtype = "f64")
    out <- prim_qr(A)
    expect_named(out, c("Q", "R"))
    Q <- as_array(out$Q)
    R <- as_array(out$R)
    expect_equal(dim(Q), c(m, k))
    expect_equal(dim(R), c(k, n))
    expect_equal(Q %*% R, as_array(A), tolerance = 1e-5)
    # Documented: Q has orthonormal columns, R is upper triangular.
    expect_equal(t(Q) %*% Q, diag(k), tolerance = 1e-5)
    expect_equal(R[lower.tri(R)], rep(0, sum(lower.tri(R))))
  })

  it("rejects invalid inputs", {
    vec <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_error(prim_qr(vec), "must be a 2-D matrix")
    empty <- nv_matrix(numeric(0), nrow = 0, ncol = 2, dtype = "f32")
    expect_error(prim_qr(empty), "zero-sized")
    int_mat <- nv_matrix(1:4, nrow = 2, dtype = "i32")
    expect_error(prim_qr(int_mat), "floating-point")
  })
})

describe("prim_lu", {
  it("decomposes a square matrix", {
    A <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    out <- prim_lu(A)
    expect_named(out, c("LU", "pivots", "permutation"))
    expect_equal(shape(out$LU), c(2L, 2L))
    expect_equal(shape(out$pivots), 2L)
    expect_equal(shape(out$permutation), 2L)
    LU <- as_array(out$LU)
    pivots <- as_array(out$pivots)
    permutation <- as_array(out$permutation)
    # Documented: pivots are 1-based, each in 1..m; permutation is a 1-based
    # permutation of 1..m such that (P %*% A)[i, ] == A[permutation[i], ].
    expect_true(all(pivots >= 1L & pivots <= nrow(LU)))
    expect_setequal(permutation, seq_len(nrow(LU)))
    # Reconstruct: L unit-lower-triangular, U upper-triangular; P A == L U.
    L <- LU
    L[upper.tri(L)] <- 0
    diag(L) <- 1
    U <- LU
    U[lower.tri(U)] <- 0
    A_mat <- as_array(A)
    expect_equal(L %*% U, A_mat[permutation, , drop = FALSE], tolerance = 1e-5)
  })

  it("rejects invalid inputs", {
    vec <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_error(prim_lu(vec), "must be a 2-D matrix")
    empty <- nv_matrix(numeric(0), nrow = 0, ncol = 2, dtype = "f32")
    expect_error(prim_lu(empty), "zero-sized")
    int_mat <- nv_matrix(1:4, nrow = 2, dtype = "i32")
    expect_error(prim_lu(int_mat), "floating-point")
  })
})

describe("prim_svd", {
  expect_thin_svd <- function(A) {
    m <- nrow(A)
    n <- ncol(A)
    k <- min(m, n)
    out <- prim_svd(A)
    expect_named(out, c("d", "u", "vt"))
    d <- as_array(out$d)
    u <- as_array(out$u)
    vt <- as_array(out$vt)
    expect_equal(dim(u), c(m, k))
    expect_equal(length(d), k)
    expect_equal(dim(vt), c(k, n))
    expect_equal(u %*% diag(d) %*% vt, as_array(A), tolerance = 1e-5)
    expect_true(all(d >= 0))
    expect_equal(d, sort(d, decreasing = TRUE))
    expect_equal(t(u) %*% u, diag(k), tolerance = 1e-5)
    expect_equal(vt %*% t(vt), diag(k), tolerance = 1e-5)
  }

  it("decomposes a tall matrix", {
    # Returns `vt` (LAPACK form, shape (k, n)) — not `v` like base::svd().
    A <- nv_matrix(c(1, 0, 0, 1, 0, 1), nrow = 3L, dtype = "f64")
    expect_thin_svd(A)
  })

  # On CUDA this exercises the layout-flip path in prim_svd[["stablehlo"]]
  # (transposed = TRUE + row-major x / U / Vt layouts), which the
  # cuSOLVER FFI handler resolves by swapping m / n and the U / Vt slots.
  it("decomposes a wide matrix", {
    A <- nv_matrix(c(1, 0, 0, 1, 0, 1), nrow = 2L, dtype = "f64")
    expect_thin_svd(A)
  })

  it("rejects invalid inputs", {
    vec <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_error(prim_svd(vec), "must be a 2-D matrix")
    empty <- nv_matrix(numeric(0), nrow = 0, ncol = 2, dtype = "f32")
    expect_error(prim_svd(empty), "zero-sized")
    int_mat <- nv_matrix(1:4, nrow = 2, dtype = "i32")
    expect_error(prim_svd(int_mat), "floating-point")
  })
})

describe("prim_eigh", {
  it("decomposes a symmetric matrix", {
    A <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    out <- prim_eigh(A)
    # Names and order match base::eigen(): values, vectors.
    expect_named(out, c("values", "vectors"))
    values <- as_array(out$values)
    vectors <- as_array(out$vectors)
    expect_equal(length(values), 2L)
    expect_equal(dim(vectors), c(2L, 2L))
    expect_equal(
      vectors %*% diag(values) %*% t(vectors),
      as_array(A),
      tolerance = 1e-5
    )
    # Documented: values in ascending order; vectors has orthonormal columns.
    expect_equal(values, sort(values))
    expect_equal(t(vectors) %*% vectors, diag(ncol(vectors)), tolerance = 1e-5)
  })

  it("rejects invalid inputs", {
    vec <- nv_array(c(1, 2, 3), dtype = "f32")
    expect_error(prim_eigh(vec), "must be a 2-D matrix")
    empty <- nv_matrix(numeric(0), nrow = 0, ncol = 0, dtype = "f32")
    expect_error(prim_eigh(empty), "zero-sized")
    int_mat <- nv_matrix(1:4, nrow = 2, dtype = "i32")
    expect_error(prim_eigh(int_mat), "floating-point")
    rect <- nv_matrix(1:6, nrow = 2, dtype = "f32")
    expect_error(prim_eigh(rect), "square")
  })
})


test_that("error when multiplying lists in if-statement", {
  f <- jit(function(pred, x) {
    prim_if(pred, \() x + x, \() x * x)
  })
  expect_error(
    f(nv_scalar(FALSE), list(nv_scalar(2))),
    "non-numeric argument to binary operator"
  )
})

test_that("prim_is_finite", {
  x <- nv_array(c(1.0, Inf, -Inf, NaN), dtype = "f32")
  expect_equal(prim_is_finite(x), nv_array(c(TRUE, FALSE, FALSE, FALSE), dtype = "bool"))
})

test_that("prim_clamp", {
  x <- nv_array(c(-2.0, -0.5, 0.5, 2.0), dtype = "f32")
  min_val <- nv_broadcast_to(nv_scalar(-1.0, "f32"), shape(x))
  max_val <- nv_broadcast_to(nv_scalar(1.0, "f32"), shape(x))
  expect_equal(prim_clamp(min_val, x, max_val), nv_array(c(-1.0, -0.5, 0.5, 1.0), dtype = "f32"))
})

test_that("prim_reverse", {
  x <- nv_array(1:5, dtype = "i32")
  expect_equal(prim_reverse(x, 1L), nv_array(5:1, dtype = "i32"))

  # 2D reverse
  x2 <- nv_matrix(1:6, nrow = 2, ncol = 3, dtype = "i32")
  expect_equal(prim_reverse(x2, 2L), nv_matrix(c(5L, 6L, 3L, 4L, 1L, 2L), nrow = 2, ncol = 3, dtype = "i32"))
})

test_that("prim_iota", {
  expect_equal(prim_iota(1L, "i32", 5L, start = 0L), nv_array(0:4, dtype = "i32"))

  expect_equal(prim_iota(1L, "i32", 5L, start = 1L), nv_array(1:5, dtype = "i32"))

  # 2D along first dimension (default start = 1)
  expected <- matrix(c(1L, 2L, 3L, 1L, 2L, 3L), 3, 2)
  expect_equal(prim_iota(1L, "i32", c(3L, 2L)), nv_array(expected, dtype = "i32"))
})

test_that("prim_popcnt", {
  x <- nv_array(c(0L, 1L, 2L, 3L, 7L, 255L), dtype = "i32")
  expect_equal(prim_popcnt(x), nv_array(c(0L, 1L, 1L, 2L, 3L, 8L), dtype = "i32"))
})

test_that("prim_gather", {
  # Simple 1D gather: select elements at specific indices
  x <- nv_array(c(10L, 20L, 30L, 40L, 50L), dtype = "i32")
  indices <- nv_array(c(1L, 3L, 5L), dtype = "i64", shape = c(3, 1))
  out <- prim_gather(
    x = x,
    start_indices = indices,
    slice_sizes = c(1L),
    offset_dims = integer(),
    collapsed_slice_dims = 1L,
    x_batching_dims = integer(),
    start_indices_batching_dims = integer(),
    start_index_map = 1L,
    index_vector_dim = 2L,
    indices_are_sorted = FALSE,
    unique_indices = FALSE
  )
  expect_equal(out, nv_array(c(10L, 30L, 50L), dtype = "i32"))
})

test_that("prim_scatter", {
  # Simple 1D scatter: update elements at specific indices
  f <- jit(function(x, indices, updates) {
    prim_scatter(
      x = x,
      scatter_indices = indices,
      update = updates,
      update_window_dims = integer(),
      inserted_window_dims = 1L,
      x_batching_dims = integer(),
      scatter_indices_batching_dims = integer(),
      scatter_dims_to_x_dims = 1L,
      index_vector_dim = 2L,
      indices_are_sorted = FALSE,
      unique_indices = TRUE,
      update_computation = function(old, new) new
    )
  })

  x <- nv_array(c(1L, 2L, 3L, 4L, 5L), dtype = "i32")
  indices <- nv_array(c(1L, 3L, 5L), dtype = "i64", shape = c(3, 1))
  updates <- nv_array(c(100L, 300L, 500L), dtype = "i32")
  out <- f(x, indices, updates)
  expect_equal(out, nv_array(c(100L, 2L, 300L, 4L, 500L), dtype = "i32"))
})

test_that("prim_print", {
  f <- jit(function(x) prim_print(x))
  x <- nv_array(c(1.0, 2.0, 3.0), dtype = "f32")
  expect_snapshot({
    out <<- f(x)
  })
  expect_equal(x, out)
})

describe("prim_sort", {
  it("sorts a 1D vector", {
    x <- nv_array(c(3, 1, 4, 2, 5))
    expect_equal(prim_sort(list(x), dim = 1L)[[1L]], nv_array(c(1, 2, 3, 4, 5)))
  })

  it("sorts each row (dim = 2) of a matrix", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(prim_sort(list(m), dim = 2L)[[1L]], nv_matrix(c(1, 3, 5, 0, 2, 4), nrow = 2, byrow = TRUE))
  })

  it("variadic: carried array is permuted by the key", {
    x <- nv_array(c(3, 1, 4, 2, 5))
    idx <- nv_iota(dim = 1L, dtype = "i64", shape = 5L)
    out <- prim_sort(list(x, idx), dim = 1L)
    expect_equal(
      out,
      list(nv_array(c(1, 2, 3, 4, 5)), nv_array(c(2L, 4L, 1L, 3L, 5L), dtype = "i64"))
    )
  })

  it("float ascending: all NaN values land at the end regardless of sign", {
    out <- as.numeric(prim_sort(list(arr(c(NaN, 1, 2, -NaN))), dim = 1L)[[1L]])
    expect_equal(out[1:2], c(1, 2))
    expect_true(all(is.nan(out[3:4])))
  })

  it("float descending: NaN values land at the beginning", {
    out <- as.numeric(
      prim_sort(list(arr(c(1, NaN, 2, -NaN))), dim = 1L, descending = TRUE)[[1L]]
    )
    expect_true(all(is.nan(out[1:2])))
    expect_equal(out[3:4], c(2, 1))
  })

  it("float ordering interleaves -Inf, finite, +Inf, NaN correctly", {
    out <- as.numeric(
      prim_sort(list(arr(c(-0, 1, -1, 0, NaN, Inf, -Inf))), dim = 1L)[[1L]]
    )
    # -Inf < -1 < {-0, +0} (tied) < 1 < +Inf < NaN
    expect_equal(out[1:2], c(-Inf, -1))
    expect_equal(out[3:4], c(0, 0))
    expect_equal(out[5:6], c(1, Inf))
    expect_true(is.nan(out[7]))
  })

  it("argsort puts NaN positions last under ascending sort", {
    x <- arr(c(1, NaN, 2, -NaN))
    idx <- nv_iota(dim = 1L, dtype = "i32", shape = 4L)
    perm <- as.integer(prim_sort(list(x, idx), dim = 1L)[[2L]])
    # values at perm[1:2] are non-NaN (positions of 1 and 2 = 1 and 3),
    # perm[3:4] are the NaN positions in some order
    expect_equal(perm[1:2], c(1L, 3L))
    expect_setequal(perm[3:4], c(2L, 4L))
  })

  it("integer keys are unaffected by the float canonicalization path", {
    x <- nv_array(c(3L, 1L, 4L, 1L, 5L))
    expect_equal(prim_sort(list(x), dim = 1L)[[1L]], nv_array(c(1L, 1L, 3L, 4L, 5L)))
  })

  it("stable sort: signed zeros and NaNs canonicalize to equal keys, so argsort is 1:n", {
    # -0/+0 collapse to +0, -NaN/+NaN collapse to +NaN, so within each group
    # all keys compare equal under TOTALORDER. With is_stable = TRUE the input
    # order must be preserved, giving indices 1:n (0s before NaNs).
    x <- arr(c(-0, 0, 0, -0, -NaN, NaN, -NaN, NaN))
    idx <- nv_iota(dim = 1L, dtype = "i32", shape = 8L)
    perm <- as.integer(prim_sort(list(x, idx), dim = 1L, is_stable = TRUE)[[2L]])
    expect_equal(perm, 1:8)
  })
})

describe("prim_top_k", {
  it("returns values and 1-based indices along the last dim", {
    out <- prim_top_k(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)), k = 3L)
    expect_length(out, 2L)
    expect_equal(as.vector(out[[1L]]), c(9, 6, 5))
    expect_equal(as.vector(out[[2L]]), c(6L, 8L, 5L))
    expect_equal(as.character(dtype(out[[2L]])), "i32")
  })

  it("operates per-row on a matrix", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    out <- prim_top_k(m, k = 2L)
    expect_equal(as_array(out[[1L]]), matrix(c(5, 3, 4, 2), nrow = 2, byrow = TRUE))
    expect_equal(as_array(out[[2L]]), matrix(c(3L, 1L, 2L, 1L), nrow = 2, byrow = TRUE))
  })

  it("breaks ties with the smaller index", {
    out <- prim_top_k(nv_array(c(1, 5, 5, 3)), k = 2L)
    expect_equal(as.vector(out[[1L]]), c(5, 5))
    expect_equal(as.vector(out[[2L]]), c(2L, 3L))
  })

  it("preserves the input dtype on values output", {
    out <- prim_top_k(nv_array(c(5L, 2L, 8L, 1L), dtype = "i32"), k = 2L)
    expect_equal(as.character(dtype(out[[1L]])), "i32")
    expect_equal(as.vector(out[[1L]]), c(8L, 5L))
  })

  it("rejects k larger than the last dim", {
    expect_error(prim_top_k(nv_array(c(1, 2, 3)), k = 5L))
  })
})

describe("prim_argmax", {
  it("returns the 1-based index of the max along a 1D array", {
    expect_equal(as_array(prim_argmax(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)), dim = 1L)), 6L)
  })

  it("breaks ties with the smallest index", {
    expect_equal(as_array(prim_argmax(nv_array(c(1, 5, 5, 3)), dim = 1L)), 2L)
  })

  it("operates per-row on a matrix", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(as.vector(prim_argmax(m, dim = 2L)), c(3L, 2L))
  })

  it("supports drop = FALSE", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    out <- prim_argmax(m, dim = 2L, drop = FALSE)
    expect_equal(shape(out), c(2L, 1L))
    expect_equal(as.vector(out), c(3L, 2L))
  })

  it("returns dtype i32", {
    out <- prim_argmax(nv_array(c(1, 2, 3)), dim = 1L)
    expect_equal(as.character(dtype(out)), "i32")
  })

  it("works with integer input", {
    out <- prim_argmax(nv_array(c(5L, 2L, 8L, 1L), dtype = "i32"), dim = 1L)
    expect_equal(as_array(out), 3L)
  })

  it("errors at trace time when reducing along a size-0 axis", {
    expect_error(
      prim_argmax(nv_array(numeric(0), shape = 0L), dim = 1L),
      "undefined for an empty axis"
    )
    expect_error(
      prim_argmax(nv_matrix(numeric(0), nrow = 3, ncol = 0), dim = 2L),
      "undefined for an empty axis"
    )
    # Inside jit too.
    expect_error(
      jit(function(x) prim_argmax(x, dim = 1L))(nv_array(numeric(0), shape = 0L)),
      "undefined for an empty axis"
    )
  })

  it("permits reducing along a non-empty axis when another axis is size 0", {
    # 2D with shape (0, 3): reducing along dim 2 (size 3) is well-defined and
    # produces an empty (length-0) i32 vector.
    m <- nv_matrix(numeric(0), nrow = 0, ncol = 3)
    out <- prim_argmax(m, dim = 2L)
    expect_equal(shape(out), 0L)
    expect_equal(as.character(dtype(out)), "i32")
  })
})

describe("prim_argmin", {
  it("returns the 1-based index of the min along a 1D array", {
    expect_equal(as_array(prim_argmin(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)), dim = 1L)), 2L)
  })

  it("breaks ties with the smallest index", {
    expect_equal(as_array(prim_argmin(nv_array(c(3, 1, 4, 1, 5)), dim = 1L)), 2L)
  })

  it("operates per-column on a matrix (dim = 1)", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(as.vector(prim_argmin(m, dim = 1L)), c(2L, 1L, 2L))
  })

  it("errors at trace time when reducing along a size-0 axis", {
    expect_error(
      prim_argmin(nv_array(numeric(0), shape = 0L), dim = 1L),
      "undefined for an empty axis"
    )
  })
})

describe("prim_reduce", {
  it("sum via prim_add", {
    out <- prim_reduce(nv_array(c(1, 2, 3, 4)), init = nv_scalar(0), dims = 1L, reductor = prim_add)
    expect_equal(as_array(out), 10)
  })

  it("product via prim_mul", {
    out <- prim_reduce(nv_array(c(1, 2, 3, 4)), init = nv_scalar(1), dims = 1L, reductor = prim_mul)
    expect_equal(as_array(out), 24)
  })

  it("custom max via prim_max with -Inf init", {
    out <- prim_reduce(nv_array(c(3, 1, 4, 1, 5, 9, 2)), init = nv_scalar(-Inf), dims = 1L, reductor = prim_max)
    expect_equal(as_array(out), 9)
  })

  it("supports drop = FALSE", {
    m <- nv_matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
    out <- prim_reduce(m, init = nv_scalar(0), dims = 2L, drop = FALSE, reductor = prim_add)
    expect_equal(shape(out), c(2L, 1L))
    expect_equal(as.vector(out), c(9, 12))
  })

  it("rejects mismatched init dtype", {
    expect_error(
      prim_reduce(nv_array(c(1, 2, 3)), init = nv_scalar(0L, dtype = "i32"), dims = 1L, reductor = prim_add),
      "same dtype"
    )
  })

  it("rejects non-scalar init", {
    expect_error(
      prim_reduce(nv_array(c(1, 2, 3)), init = nv_array(c(0, 0)), dims = 1L, reductor = prim_add),
      "scalar"
    )
  })
})

test_that("prim_dot_general precision", {
  A <- nv_matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, dtype = "f32")
  B <- nv_matrix(c(6, 5, 4, 3, 2, 1), nrow = 3, ncol = 2, dtype = "f32")
  ref <- as_array(A) %*% as_array(B)

  for (prec in c("default", "high", "highest")) {
    out <- jit(function(a, b) {
      prim_dot_general(
        a,
        b,
        contracting_dims = list(2L, 1L),
        batching_dims = list(integer(), integer()),
        precision = prec
      )
    })(A, B)
    expect_equal(as_array(out), ref, tolerance = 1e-4)
  }

  # the chosen precision is lowered into the StableHLO IR (uppercased)
  g <- trace_fn(
    function(a, b) {
      prim_dot_general(
        a,
        b,
        contracting_dims = list(2L, 1L),
        batching_dims = list(integer(), integer()),
        precision = "high"
      )
    },
    list(nv_aval("f32", shape = c(2, 3)), nv_aval("f32", shape = c(3, 2)))
  )
  expect_match(repr(stablehlo(g)[[1L]]), "precision = [HIGH, HIGH]", fixed = TRUE)

  # the default precision is "highest"
  g_default <- trace_fn(
    function(a, b) {
      prim_dot_general(
        a,
        b,
        contracting_dims = list(2L, 1L),
        batching_dims = list(integer(), integer())
      )
    },
    list(nv_aval("f32", shape = c(2, 3)), nv_aval("f32", shape = c(3, 2)))
  )
  expect_match(repr(stablehlo(g_default)[[1L]]), "precision = [HIGHEST, HIGHEST]", fixed = TRUE)

  # invalid precision is rejected
  expect_error(
    prim_dot_general(
      A,
      B,
      contracting_dims = list(2L, 1L),
      batching_dims = list(integer(), integer()),
      precision = "bogus"
    ),
    "should be one of"
  )
})

test_that("nv_matmul exposes precision and defaults to highest", {
  A <- nv_matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, dtype = "f32")
  B <- nv_matrix(c(6, 5, 4, 3, 2, 1), nrow = 3, ncol = 2, dtype = "f32")
  ref <- as_array(A) %*% as_array(B)
  expect_equal(as_array(nv_matmul(A, B, precision = "high")), ref, tolerance = 1e-4)

  g <- trace_fn(
    function(a, b) nv_matmul(a, b),
    list(nv_aval("f32", shape = c(2, 3)), nv_aval("f32", shape = c(3, 2)))
  )
  expect_match(repr(stablehlo(g)[[1L]]), "precision = [HIGHEST, HIGHEST]", fixed = TRUE)
})

# we don't want to include torch in Suggests just for the tests, as it's a relatively
# heavy dependency
# We have a CI job that installs torch, so it's at least tested once

if (nzchar(system.file(package = "torch"))) {
  source(system.file("extra-tests", "test-primitives-stablehlo-torch.R", package = "anvl"), local = TRUE)
}
