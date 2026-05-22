test_that("auto-broadcasting higher-dimensional arrays is not supported (it's bug prone)", {
  x <- nv_array(1:2, shape = c(2, 1))
  y <- nv_array(1:2, shape = c(1, 2))
  expect_error(
    nv_add(x, y),
    "must have the same shape"
  )
})

test_that("nv_fill rejects non-scalar-R value with a helpful message", {
  x <- nv_scalar(1, dtype = "f32")
  expect_error(
    nv_fill(x, shape = c(2, 3)),
    "must be an R vector of length 1"
  )
  expect_error(
    nv_fill(c(1, 2), shape = c(2, 3)),
    "must be an R vector of length 1"
  )
  expect_error(
    nv_fill("a", shape = c(2, 3)),
    "must be an R vector of length 1"
  )
})

test_that("broadcasting scalars", {
  expect_equal(
    nv_add(
      nv_scalar(1),
      nv_array(0, shape = c(2, 2))
    ),
    nv_array(1, shape = c(2, 2))
  )
})

test_that("infix add", {
  expect_equal(
    nv_array(1, shape = c(2, 2)) + nv_array(0, shape = c(2, 2)),
    nv_array(1, shape = c(2, 2))
  )
})

test_that("jit constant single return is bare array", {
  f <- jit(function() nv_scalar(0.5))
  out <- f()
  expect_equal(as_array(out), 0.5, tolerance = 1e-6)
})

test_that("Summary group generics", {
  expect_equal(as_array(sum(nv_array(1:10))), 55)
})

test_that("mean", {
  expect_equal(as_array(mean(nv_array(1:10, "f32"))), 5.5)
})

test_that("constants can be lifted to the appropriate level", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_equal(
    jit(gradient(f, wrt = "x"))(nv_scalar(2))[[1L]],
    nv_scalar(1)
  )
})

test_that("wrt non-existent argument", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_error(
    jit(gradient(f, wrt = "y"))(nv_array(2)),
    "must be a subset"
  )
})

test_that("promote to common", {
  expect_equal(
    nv_add(nv_array(1, dtype = "i32"), nv_array(1.0, dtype = "f32")),
    nv_array(2.0, dtype = "f32")
  )
})

test_that("nv_clamp converts min and max to operand dtype", {
  expect_equal(
    nv_clamp(nv_scalar(0L), nv_array(c(-1, 0.5, 2), dtype = "f32"), nv_scalar(1L)),
    nv_array(c(0, 0.5, 1), dtype = "f32")
  )
})

test_that("nv_ifelse broadcasts scalars and promotes branches to a common dtype", {
  pred <- nv_array(c(TRUE, FALSE, TRUE))
  expect_equal(
    nv_ifelse(pred, nv_scalar(1L), nv_array(c(0.5, 0.5, 0.5), dtype = "f32")),
    nv_array(c(1, 0.5, 1), dtype = "f32")
  )
})

describe("nv_concatenate", {
  it("auto-promotes to common", {
    expect_equal(
      nv_concatenate(nv_array(c(1, 2)), nv_array(3:4)),
      nv_array(c(1, 2, 3, 4))
    )
  })
  it("can concatenate literals", {
    # Pure literals produce ambiguous output
    expect_equal(
      nv_concatenate(1L, 2L),
      nv_array(1:2, ambiguous = TRUE)
    )
    expect_equal(
      nv_concatenate(1L, 2L, dimension = 1L),
      nv_array(1:2, ambiguous = TRUE)
    )
    # Mixed array + literal: non-ambiguous array determines output ambiguity
    expect_equal(
      nv_concatenate(nv_array(1:2), 3L),
      nv_array(1:3)
    )
    expect_equal(
      nv_concatenate(nv_array(1L), 2L),
      nv_array(1:2)
    )
  })
  it("fails when dimension is out of bounds", {
    expect_error(
      nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), dimension = 3L)
    )
  })
  it("can concatenate 2d arrays", {
    expect_equal(
      nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), dimension = 1L),
      nv_array(1:4, shape = c(4, 1))
    )
    expect_equal(
      nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), dimension = 2L),
      nv_array(1:4, shape = c(2, 2), dtype = "i32")
    )
  })
  it("fails with incompatible shapes", {
    expect_error(
      nv_concatenate(nv_array(1, shape = c(1, 1, 1)), nv_array(2, shape = c(1, 1)), dimension = 1L)
    )
  })
})

describe("nv_rbind", {
  it("stacks two 1-D vectors as rows (eager)", {
    x <- nv_array(c(1, 2, 3))
    y <- nv_array(c(4, 5, 6))
    expect_equal(
      rbind(x, y),
      nv_array(rbind(c(1, 2, 3), c(4, 5, 6)))
    )
  })

  it("stacks two 1-D vectors as rows (jit)", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3))
        y <- nv_array(c(4, 5, 6))
        rbind(x, y)
      },
      nv_array(rbind(c(1, 2, 3), c(4, 5, 6)))
    )
  })

  it("stacks two matrices vertically", {
    a <- matrix(1:6, nrow = 2)
    b <- matrix(7:12, nrow = 2)
    expect_equal(
      {
        x <- nv_array(a)
        y <- nv_array(b)
        rbind(x, y)
      },
      nv_array(rbind(a, b))
    )
  })

  it("treats 1-D operand as a row when mixed with a matrix", {
    a <- matrix(1:6, nrow = 2)
    v <- c(7, 8, 9)
    expect_equal(
      {
        x <- nv_array(a)
        y <- nv_array(v)
        rbind(x, y)
      },
      nv_array(rbind(a, v))
    )
  })

  it("accepts more than two arguments", {
    expect_equal(
      {
        rbind(nv_array(c(1, 2)), nv_array(c(3, 4)), nv_array(c(5, 6)))
      },
      nv_array(rbind(c(1, 2), c(3, 4), c(5, 6)))
    )
  })

  it("promotes operands to a common dtype", {
    x <- nv_array(c(1L, 2L, 3L))
    y <- nv_array(c(4, 5, 6))
    out <- rbind(x, y)
    expect_equal(dtype(out), dtype(y))
  })

  it("errors when number of columns mismatch", {
    expect_error(rbind(nv_array(c(1, 2)), nv_array(c(3, 4, 5))))
  })

  it("nv_rbind matches rbind", {
    x <- nv_array(c(1, 2, 3))
    y <- nv_array(c(4, 5, 6))
    expect_equal(nv_rbind(x, y), rbind(x, y))
  })

  it("broadcasts a scalar to the column count of a matrix", {
    a <- matrix(1:6, nrow = 2)
    expect_equal(rbind(nv_array(a), nv_scalar(0)), nv_array(rbind(a, 0)))
    expect_equal(rbind(nv_scalar(0), nv_array(a)), nv_array(rbind(0, a)))
  })

  it("broadcasts a scalar to the column count of a 1-D vector", {
    v <- c(7, 8, 9)
    expect_equal(rbind(nv_array(v), nv_scalar(0)), nv_array(rbind(v, 0)))
  })

  it("treats all scalars as 1x1 rows", {
    expect_equal(rbind(nv_scalar(1), nv_scalar(2)), nv_array(rbind(1, 2)))
  })

  it("stacks two 3-D arrays along dimension 1", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    b <- array(101:124, dim = c(2L, 3L, 4L))
    out <- rbind(nv_array(a), nv_array(b))
    expect_equal(shape(out), c(4L, 3L, 4L))
    expect_equal(as_array(out)[1:2, , ], a)
    expect_equal(as_array(out)[3:4, , ], b)
  })

  it("broadcasts a scalar against a 3-D array", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    out <- rbind(nv_array(a), nv_scalar(0))
    expect_equal(shape(out), c(3L, 3L, 4L))
    expect_equal(as_array(out)[1:2, , ], a)
    expect_equal(as_array(out)[3L, , ], array(0, dim = c(3L, 4L)))
  })
})

describe("nv_cbind", {
  it("stacks two 1-D vectors as columns (eager)", {
    x <- nv_array(c(1, 2, 3))
    y <- nv_array(c(4, 5, 6))
    expect_equal(
      cbind(x, y),
      nv_array(cbind(c(1, 2, 3), c(4, 5, 6)))
    )
  })

  it("stacks two 1-D vectors as columns (jit)", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3))
        y <- nv_array(c(4, 5, 6))
        cbind(x, y)
      },
      nv_array(cbind(c(1, 2, 3), c(4, 5, 6)))
    )
  })

  it("treats 1-D operand as a column when mixed with a matrix", {
    a <- matrix(1:6, nrow = 3)
    v <- c(7, 8, 9)
    expect_equal(
      {
        x <- nv_array(a)
        y <- nv_array(v)
        cbind(x, y)
      },
      nv_array(cbind(a, v))
    )
  })

  it("broadcasts a scalar to the row count of a 1-D vector", {
    v <- c(7, 8, 9)
    expect_equal(cbind(nv_array(v), nv_scalar(0)), nv_array(cbind(v, 0)))
  })

  it("stacks two matrices horizontally", {
    a <- matrix(1:6, nrow = 3)
    b <- matrix(7:12, nrow = 3)
    expect_equal(
      {
        x <- nv_array(a)
        y <- nv_array(b)
        cbind(x, y)
      },
      nv_array(cbind(a, b))
    )
  })

  it("accepts more than two arguments", {
    a <- matrix(c(1, 2), ncol = 1L)
    b <- matrix(c(3, 4), ncol = 1L)
    c <- matrix(c(5, 6), ncol = 1L)
    expect_equal(
      cbind(nv_array(a), nv_array(b), nv_array(c)),
      nv_array(cbind(a, b, c))
    )
  })

  it("errors when number of rows mismatch", {
    expect_error(cbind(
      nv_matrix(c(1, 2), ncol = 1L),
      nv_matrix(c(3, 4, 5), ncol = 1L)
    ))
  })

  it("nv_cbind matches cbind", {
    x <- nv_matrix(1:3, ncol = 1L)
    y <- nv_matrix(4:6, ncol = 1L)
    expect_equal(nv_cbind(x, y), cbind(x, y))
  })

  it("broadcasts a scalar to the row count of a matrix", {
    a <- matrix(1:6, nrow = 3)
    expect_equal(cbind(nv_array(a), nv_scalar(0)), nv_array(cbind(a, 0)))
    expect_equal(cbind(nv_scalar(0), nv_array(a)), nv_array(cbind(0, a)))
  })

  it("treats all scalars as 1x1 columns", {
    expect_equal(cbind(nv_scalar(1), nv_scalar(2)), nv_array(cbind(1, 2)))
  })

  it("stacks two 3-D arrays along dimension 2", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    b <- array(101:124, dim = c(2L, 3L, 4L))
    out <- cbind(nv_array(a), nv_array(b))
    expect_equal(shape(out), c(2L, 6L, 4L))
    expect_equal(as_array(out)[, 1:3, ], a)
    expect_equal(as_array(out)[, 4:6, ], b)
  })

  it("broadcasts a scalar against a 3-D array", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    out <- cbind(nv_array(a), nv_scalar(0))
    expect_equal(shape(out), c(2L, 4L, 4L))
    expect_equal(as_array(out)[, 1:3, ], a)
    expect_equal(as_array(out)[, 4L, ], array(0, dim = c(2L, 4L)))
  })
})

describe("nv_log2", {
  it("computes base-2 logarithm", {
    expect_equal(
      nv_log2(nv_array(c(1, 2, 4, 8))),
      nv_array(log2(c(1, 2, 4, 8))),
      tolerance = 1e-6
    )
  })
  it("works on scalars", {
    expect_equal(
      nv_log2(nv_scalar(16)),
      nv_scalar(4),
      tolerance = 1e-6
    )
  })
})

describe("nv_log10", {
  it("computes base-10 logarithm", {
    expect_equal(
      nv_log10(nv_array(c(1, 10, 100, 1000))),
      nv_array(log10(c(1, 10, 100, 1000))),
      tolerance = 1e-6
    )
  })
  it("works on scalars", {
    expect_equal(
      nv_log10(nv_scalar(1000)),
      nv_scalar(3),
      tolerance = 1e-6
    )
  })
})

describe("nv_is_finite", {
  it("detects finite values", {
    expect_equal(
      nv_is_finite(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(TRUE, FALSE, FALSE, FALSE, TRUE))
    )
  })
  it("works via is.finite() generic", {
    expect_equal(
      is.finite(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(TRUE, FALSE, FALSE, FALSE, TRUE))
    )
  })
})

describe("nv_is_nan", {
  it("detects NaN values", {
    expect_equal(
      nv_is_nan(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(FALSE, TRUE, FALSE, FALSE, FALSE))
    )
  })
  it("works via is.nan() generic", {
    expect_equal(
      is.nan(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(FALSE, TRUE, FALSE, FALSE, FALSE))
    )
  })
})

describe("nv_is_infinite", {
  it("detects infinite values", {
    expect_equal(
      nv_is_infinite(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(FALSE, FALSE, TRUE, TRUE, FALSE))
    )
  })
  it("works via is.infinite() generic", {
    expect_equal(
      is.infinite(nv_array(c(1, NaN, Inf, -Inf, 0))),
      nv_array(c(FALSE, FALSE, TRUE, TRUE, FALSE))
    )
  })
})

describe("nv_var", {
  it("computes variance with Bessel's correction", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_equal(
      nv_var(nv_array(vals), dims = 1L),
      nv_scalar(var(vals)),
      tolerance = 1e-5
    )
  })
  it("computes population variance with correction = 0", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expected <- mean((vals - mean(vals))^2)
    expect_equal(
      nv_var(nv_array(vals), dims = 1L, correction = 0L),
      nv_scalar(expected),
      tolerance = 1e-5
    )
  })
  it("works along specific dimensions of a matrix", {
    vals <- c(1, 2, 3, 4, 5, 6)
    m <- matrix(vals, nrow = 2)
    expect_equal(
      nv_var(nv_array(vals, shape = c(2, 3), dtype = "f32"), dims = 2L),
      nv_array(apply(m, 1, var), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("reduces over all axes by default", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_equal(
      nv_var(nv_array(vals)),
      nv_scalar(var(vals)),
      tolerance = 1e-5
    )
    expect_equal(
      nv_var(nv_array(vals), dims = NULL),
      nv_scalar(var(vals)),
      tolerance = 1e-5
    )
  })
  it("rejects out-of-range and duplicate dims", {
    expect_error(nv_var(nv_array(c(1, 2, 3, 4)), dims = 5L))
    expect_error(nv_var(nv_array(c(1, 2, 3, 4)), dims = c(1L, 1L)))
  })
})

describe("nv_sd", {
  it("computes standard deviation", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_equal(
      nv_sd(nv_array(vals), dims = 1L),
      nv_scalar(sd(vals)),
      tolerance = 1e-5
    )
  })
  it("reduces over all axes by default", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_equal(
      nv_sd(nv_array(vals)),
      nv_scalar(sd(vals)),
      tolerance = 1e-5
    )
    expect_equal(
      nv_sd(nv_array(vals), dims = NULL),
      nv_scalar(sd(vals)),
      tolerance = 1e-5
    )
  })
})

describe("nv_squeeze", {
  it("removes all size-1 dimensions by default", {
    expect_equal(
      {
        x <- nv_array(1:6, shape = c(1, 6, 1))
        nv_squeeze(x)
      },
      nv_array(1:6, shape = 6L)
    )
  })
  it("removes specific dimensions", {
    expect_equal(
      {
        x <- nv_array(1:6, shape = c(1, 6, 1))
        nv_squeeze(x, dims = 1L)
      },
      nv_array(1:6, shape = c(6, 1))
    )
  })
  it("errors when squeezing non-1 dimension", {
    expect_error(
      nv_squeeze(nv_array(1:6, shape = c(2, 3)), dims = 1L),
      "Cannot squeeze"
    )
  })
  it("rejects duplicate dims", {
    expect_error(
      nv_squeeze(nv_array(1:6, shape = c(1, 6, 1)), dims = c(1L, 1L))
    )
  })
})

describe("nv_unsqueeze", {
  it("adds dimension at the beginning", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3))
        nv_unsqueeze(x, dim = 1L)
      },
      nv_array(c(1, 2, 3), shape = c(1, 3))
    )
  })
  it("adds dimension at the end", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3))
        nv_unsqueeze(x, dim = 2L)
      },
      nv_array(c(1, 2, 3), shape = c(3, 1))
    )
  })
  it("adds dimension in the middle", {
    x <- nv_array(1:6, shape = c(2, 3))
    result <- nv_unsqueeze(x, dim = 2L)
    expect_equal(shape(result), c(2L, 1L, 3L))
    roundtrip <- nv_squeeze(nv_unsqueeze(x, dim = 2L), dims = 2L)
    expect_equal(roundtrip, x)
  })
})

describe("nv_seq with steps", {
  it("creates evenly spaced values", {
    expect_equal(
      nv_seq(0, 1, steps = 5L),
      nv_array(c(0, 0.25, 0.5, 0.75, 1)),
      tolerance = 1e-6
    )
  })
  it("handles single step", {
    expect_equal(
      nv_seq(3, 7, steps = 1L),
      nv_array(3, shape = 1L),
      tolerance = 1e-6
    )
  })
  it("works with integer-like endpoints", {
    expect_equal(
      nv_seq(0, 10, steps = 6L),
      nv_array(c(0, 2, 4, 6, 8, 10)),
      tolerance = 1e-6
    )
  })
})

describe("nv_outer", {
  it("computes outer product", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3))
        y <- nv_array(c(4, 5))
        nv_outer(x, y)
      },
      nv_array(c(4, 8, 12, 5, 10, 15), shape = c(3, 2)),
      tolerance = 1e-6
    )
  })
  it("promotes types", {
    expect_equal(
      {
        x <- nv_array(c(1L, 2L))
        y <- nv_array(c(1.5, 2.5))
        nv_outer(x, y)
      },
      nv_array(c(1.5, 3, 2.5, 5), shape = c(2, 2)),
      tolerance = 1e-6
    )
  })
})

describe("nv_extract_diag", {
  it("extracts diagonal from square matrix", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6, 7, 8, 9), shape = c(3, 3), dtype = "f32")
        nv_extract_diag(x)
      },
      nv_array(c(1, 5, 9), dtype = "f32")
    )
  })
  it("extracts diagonal from rectangular matrix", {
    expect_equal(
      {
        x <- nv_array(1:6, shape = c(2, 3), dtype = "f32")
        nv_extract_diag(x)
      },
      nv_array(c(1, 4), dtype = "f32")
    )
  })
})

describe("nv_trace", {
  it("computes trace of a matrix", {
    expect_equal(
      nv_trace(nv_array(c(1, 0, 0, 0, 2, 0, 0, 0, 3), shape = c(3, 3))),
      nv_scalar(6),
      tolerance = 1e-6
    )
  })
  it("computes trace of identity", {
    expect_equal(
      nv_trace(nv_eye(4L)),
      nv_scalar(4),
      tolerance = 1e-6
    )
  })
})

describe("nv_diag", {
  it("builds a diagonal matrix from a 1-D array", {
    result <- nv_diag(nv_array(c(1, 2, 3)))
    expected <- diag(c(1, 2, 3))
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("works under jit (device() on a GraphBox operand)", {
    f <- jit(function(x) nv_diag(x))
    expect_equal(
      as_array(f(nv_array(c(1, 2, 3)))),
      diag(c(1, 2, 3)),
      tolerance = 1e-6
    )
  })
  it("rejects non-1-D inputs", {
    expect_error(nv_diag(nv_matrix(1:6, nrow = 2)), "1-D array")
    expect_error(nv_diag(nv_scalar(5)), "1-D array")
  })
})

describe("nv_tril", {
  it("returns lower triangular part", {
    result <- nv_tril(nv_fill(1, c(3, 3)))
    expected <- matrix(c(1, 1, 1, 0, 1, 1, 0, 0, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("supports positive diagonal offset", {
    result <- nv_tril(nv_fill(1, c(3, 3)), diagonal = 1L)
    expected <- matrix(c(1, 1, 1, 1, 1, 1, 0, 1, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("supports negative diagonal offset", {
    result <- nv_tril(nv_fill(1, c(3, 3)), diagonal = -1L)
    expected <- matrix(c(0, 1, 1, 0, 0, 1, 0, 0, 0), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("works under jit (device() on a GraphBox operand)", {
    f <- jit(function(x) nv_tril(x, diagonal = -1L))
    expected <- matrix(c(0, 1, 1, 0, 0, 1, 0, 0, 0), nrow = 3, ncol = 3)
    expect_equal(as_array(f(nv_fill(1, c(3, 3)))), expected, tolerance = 1e-6)
  })
})

describe("nv_triu", {
  it("returns upper triangular part", {
    result <- nv_triu(nv_fill(1, c(3, 3)))
    expected <- matrix(c(1, 0, 0, 1, 1, 0, 1, 1, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("supports positive diagonal offset", {
    result <- nv_triu(nv_fill(1, c(3, 3)), diagonal = 1L)
    expected <- matrix(c(0, 0, 0, 1, 0, 0, 1, 1, 0), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("supports negative diagonal offset", {
    result <- nv_triu(nv_fill(1, c(3, 3)), diagonal = -1L)
    expected <- matrix(c(1, 1, 0, 1, 1, 1, 1, 1, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
  it("works under jit", {
    f <- jit(function(x) nv_triu(x, diagonal = 1L))
    expected <- matrix(c(0, 0, 0, 1, 0, 0, 1, 1, 0), nrow = 3, ncol = 3)
    expect_equal(as_array(f(nv_fill(1, c(3, 3)))), expected, tolerance = 1e-6)
  })
})

describe("nv_tril with quickr backend", {
  it("works when operand is quickr", {
    skip_if_no_quickr()
    x <- nv_matrix(1, nrow = 3, ncol = 3, backend = "quickr")
    result <- nv_tril(x)
    expected <- matrix(c(1, 1, 1, 0, 1, 1, 0, 0, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
})

describe("nv_triu with quickr backend", {
  it("works when operand is quickr", {
    skip_if_no_quickr()
    x <- nv_matrix(1, nrow = 3, ncol = 3, backend = "quickr")
    result <- nv_triu(x)
    expected <- matrix(c(1, 0, 0, 1, 1, 0, 1, 1, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
})

describe("nv_crossprod", {
  it("computes t(x) %*% y", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        y <- nv_array(c(7, 8, 9, 10, 11, 12), shape = c(3, 2), dtype = "f32")
        nv_crossprod(x, y)
      },
      nv_array(as.numeric(crossprod(matrix(1:6, 3, 2), matrix(7:12, 3, 2))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("computes t(x) %*% x when y is NULL", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        nv_crossprod(x)
      },
      nv_array(as.numeric(crossprod(matrix(1:6, 3, 2))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("works via S3 generic", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(3, 2), dtype = "f32")
        crossprod(x)
      },
      nv_array(as.numeric(crossprod(matrix(1:6, 3, 2))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})

describe("nv_tcrossprod", {
  it("computes x %*% t(y)", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        y <- nv_array(c(7, 8, 9, 10, 11, 12), shape = c(2, 3), dtype = "f32")
        nv_tcrossprod(x, y)
      },
      nv_array(as.numeric(tcrossprod(matrix(1:6, 2, 3), matrix(7:12, 2, 3))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("computes x %*% t(x) when y is NULL", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        nv_tcrossprod(x)
      },
      nv_array(as.numeric(tcrossprod(matrix(1:6, 2, 3))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
  it("works via S3 generic", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3, 4, 5, 6), shape = c(2, 3), dtype = "f32")
        tcrossprod(x)
      },
      nv_array(as.numeric(tcrossprod(matrix(1:6, 2, 3))), shape = c(2, 2), dtype = "f32"),
      tolerance = 1e-5
    )
  })
})

describe("nv_fill_like", {
  it("inherits shape, dtype, ambiguous, device from like", {
    like <- nv_matrix(1:6, nrow = 2, dtype = "i16")
    out <- nv_fill_like(like, 0L)
    expect_equal(shape(out), shape(like))
    expect_equal(dtype(out), dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
    expect_equal(as.integer(as_array(out)), rep(0L, 6L))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_matrix(1:6, nrow = 2, dtype = "i16")
    out <- nv_fill_like(like, 1, shape = 5L, dtype = "f32")
    expect_equal(shape(out), 5L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_iota_like", {
  it("inherits shape, dtype, ambiguous, device from like", {
    like <- nv_fill(0L, shape = c(2, 3), dtype = "i16")
    out <- nv_iota_like(like, dim = 1L)
    expect_equal(shape(out), shape(like))
    expect_equal(dtype(out), dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_fill(0L, shape = c(2, 3), dtype = "i16")
    out <- nv_iota_like(like, dim = 1L, shape = 4L, dtype = "i32")
    expect_equal(shape(out), 4L)
    expect_equal(dtype(out), as_dtype("i32"))
  })
})

describe("nv_seq_like", {
  it("inherits dtype, ambiguous, device from like (length determined by start/end)", {
    like <- nv_array(c(0L, 0L, 0L), dtype = "i16")
    out <- nv_seq_like(like, 1L, 5L)
    expect_equal(dtype(out), dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
    expect_equal(shape(out), 5L)
    expect_equal(as.integer(as_array(out)), c(1L, 2L, 3L, 4L, 5L))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_array(c(0L, 0L, 0L), dtype = "i16")
    out <- nv_seq_like(like, 1, 5, dtype = "f32")
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_select", {
  it("selects a row of a matrix and drops the dim", {
    m <- nv_matrix(1:6, nrow = 2)
    expect_equal(nv_select(m, dim = 1L, index = 1L), nv_array(c(1L, 3L, 5L)))
    expect_equal(nv_select(m, dim = 1L, index = 2L), nv_array(c(2L, 4L, 6L)))
  })

  it("selects a column of a matrix and drops the dim", {
    m <- nv_matrix(1:6, nrow = 2)
    expect_equal(nv_select(m, dim = 2L, index = 2L), nv_array(c(3L, 4L)))
  })

  it("array(i) keeps the dim with size 1", {
    x <- nv_array(1:6, shape = c(2L, 3L))
    out <- nv_select(x, dim = 2L, index = array(1L))
    expect_equal(shape(out), c(2L, 1L))
  })

  it("works on a 3D array", {
    arr <- nv_array(1:24, shape = c(2, 3, 4))
    out <- nv_select(arr, dim = 3L, index = 2L)
    expect_equal(shape(out), c(2L, 3L))
    expect_equal(as_array(out), array(7:12, dim = c(2, 3)))
  })

  it("errors when dim is out of bounds", {
    expect_error(nv_select(nv_array(c(1, 2, 3)), dim = 2L, index = 1L))
  })

  it("errors when index is out of bounds", {
    expect_error(nv_select(nv_array(c(1, 2, 3)), dim = 1L, index = 5L))
  })

  it("errors on a 0-dimensional input", {
    expect_error(nv_select(nv_scalar(1), dim = 1L, index = 1L), "0-dimensional")
  })
})

describe("nv_sort", {
  it("defaults dim to the last dimension", {
    expect_equal(
      nv_sort(nv_array(c(3, 1, 4, 1, 5))),
      nv_array(c(1, 1, 3, 4, 5))
    )
  })

  it("sorts decreasing", {
    expect_equal(
      nv_sort(nv_array(c(3, 1, 4, 1, 5)), decreasing = TRUE),
      nv_array(c(5, 4, 3, 1, 1))
    )
  })

  it("defaults to last dim for matrices (rows)", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expected <- nv_matrix(c(1, 3, 5, 0, 2, 4), nrow = 2, byrow = TRUE)
    expect_equal(nv_sort(m), expected)
  })

  it("errors on a 0-dimensional input", {
    expect_error(nv_sort(nv_scalar(1)), "0-dimensional")
  })

  it("dispatches via the sort() generic", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    expect_equal(as.vector(sort(x)), c(1, 1, 3, 4, 5))
    expect_equal(as.vector(sort(x, decreasing = TRUE)), c(5, 4, 3, 1, 1))
    expect_equal(as.vector(jit(function(x) sort(x))(x)), c(1, 1, 3, 4, 5))
  })
})

describe("nv_argsort", {
  it("returns indices that sort the array", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    perm <- as.vector(nv_argsort(x))
    expect_equal(as.vector(x)[perm], c(1, 1, 3, 4, 5))
  })

  it("supports decreasing", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    perm <- as.vector(nv_argsort(x, decreasing = TRUE))
    expect_equal(as.vector(x)[perm], c(5, 4, 3, 1, 1))
  })

  it("returns i32 dtype", {
    expect_equal(as.character(dtype(nv_argsort(nv_array(c(1, 2))))), "i32")
  })

  it("works inside jit", {
    f <- jit(function(x) nv_argsort(x))
    x <- nv_array(c(3, 1, 4, 1, 5))
    perm <- as.vector(f(x))
    expect_equal(as.vector(x)[perm], c(1, 1, 3, 4, 5))
  })
})

describe("nv_top_k", {
  it("returns the k largest values along the last dim", {
    expect_equal(
      nv_top_k(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)), k = 3L),
      nv_array(c(9, 6, 5))
    )
  })

  it("operates per-row on a matrix when dim is the last dim", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    out <- jit(nv_top_k, static = "k")(m, k = 2L)
    expect_equal(shape(out), c(2L, 2L))
    expect_equal(as_array(out), matrix(c(5, 3, 4, 2), nrow = 2, byrow = TRUE))
  })

  it("errors when k > size of dim", {
    expect_error(nv_top_k(nv_array(c(1, 2, 3)), k = 5L))
  })
})

describe("nv_median", {
  it("returns the middle element for odd length", {
    expect_equal(
      nv_median(nv_array(c(3, 1, 4, 1, 5))),
      nv_scalar(3)
    )
  })

  it("averages the two middle elements for even length", {
    expect_equal(
      nv_median(nv_array(c(1, 2, 3, 4))),
      nv_scalar(2.5)
    )
  })

  it("operates row-wise by default on a matrix", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    out <- nv_median(m)
    expect_equal(as.vector(out), c(3, 2))
  })

  it("dispatches via the median() generic", {
    expect_equal(as_array(median(nv_array(c(1, 2, 3, 4)))), as_array(nv_scalar(2.5)))
    expect_equal(
      as_array(median(nv_array(c(1, 2, 3, 4, 5)))),
      as_array(nv_scalar(3))
    )
  })

  it("forwards interpolation through nv_median and median()", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_equal(as_array(nv_median(x, interpolation = "lower")), as_array(nv_scalar(2)))
    expect_equal(as_array(nv_median(x, interpolation = "higher")), as_array(nv_scalar(3)))
    # S3 method forwards `interpolation` via `...`
    expect_equal(as_array(median(x, interpolation = "lower")), as_array(nv_scalar(2)))
    expect_equal(as_array(median(x, interpolation = "higher")), as_array(nv_scalar(3)))
  })

  it("errors when na.rm = TRUE", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_error(median(x, na.rm = TRUE), "na.rm = TRUE")
  })
})

describe("nv_quantile", {
  it("matches base R quantile (default linear / type 7) for scalar probs", {
    xr <- c(3, 1, 4, 1, 5, 9, 2, 6)
    x <- nv_array(xr)
    for (q in c(0, 0.25, 0.5, 0.75, 1)) {
      expect_equal(
        as_array(nv_quantile(x, q)),
        unname(quantile(xr, q)),
        info = paste("q =", q)
      )
    }
  })

  it("vector probs prepends a leading dim of length(probs)", {
    xr <- c(3, 1, 4, 1, 5, 9, 2, 6)
    x <- nv_array(xr)
    out <- nv_quantile(x, array(c(0.25, 0.5, 0.75)))
    expect_equal(shape(out), 3L)
    expect_equal(as.vector(out), unname(quantile(xr, c(0.25, 0.5, 0.75))))
  })

  it("vector probs work for >1-D inputs (frac broadcast)", {
    mr <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 7, 0, 5, 4), nrow = 3)
    m <- nv_array(mr)
    out <- nv_quantile(m, array(c(0.25, 0.75)))
    expect_equal(shape(out), c(2L, 3L))
    # apply(., 1, quantile) returns shape [length(probs), nrow(mr)] —
    # rows are quantile probs, cols are original rows — matching anvl's
    # leading-K layout.
    expected <- apply(mr, 1L, quantile, probs = c(0.25, 0.75))
    expect_equal(as_array(out), unname(expected), ignore_attr = TRUE)
  })

  it("interpolation = 'lower' returns sorted[floor((n-1)*q)+1]", {
    # Note: this matches NumPy's "lower" semantics; it does NOT match
    # base R's quantile(type = 1), which uses ceiling(n * q) instead.
    xr <- c(3, 1, 4, 1, 5, 9, 2, 6)
    sorted_r <- sort(xr)
    x <- nv_array(xr)
    n <- length(xr)
    for (q in c(0, 0.25, 0.4, 0.6, 0.75, 1)) {
      expected <- sorted_r[floor((n - 1) * q) + 1L]
      expect_equal(
        as_array(nv_quantile(x, q, interpolation = "lower")),
        expected,
        info = paste("q =", q)
      )
    }
  })

  it("interpolation = 'higher' picks the upper neighbour", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_equal(as_array(nv_quantile(x, 0.25, interpolation = "higher")), 2)
    expect_equal(as_array(nv_quantile(x, 0.5, interpolation = "higher")), 3)
  })

  it("interpolation = 'nearest' picks the nearer index by frac", {
    x <- nv_array(c(1, 2, 3, 4))
    # n = 4, q = 0.4 -> h = 1.2 -> lo = 2, hi = 3, frac = 0.2 < 0.5 -> lower (2)
    expect_equal(as_array(nv_quantile(x, 0.4, interpolation = "nearest")), 2)
    # q = 0.5 -> h = 1.5 -> frac = 0.5 -> NOT < 0.5 -> higher (3)
    expect_equal(as_array(nv_quantile(x, 0.5, interpolation = "nearest")), 3)
  })

  it("interpolation = 'midpoint' averages neighbours", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_equal(as_array(nv_quantile(x, 0.5, interpolation = "midpoint")), 2.5)
  })

  it("operates along a chosen dim of a matrix", {
    m_raw <- matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    m <- nv_array(m_raw)
    out <- nv_quantile(m, 0.5, dim = 2L)
    expect_equal(as.vector(out), c(3, 2))
  })

  it("rejects probs outside [0, 1]", {
    expect_error(nv_quantile(nv_array(c(1, 2)), -0.1))
    expect_error(nv_quantile(nv_array(c(1, 2)), 1.5))
  })

  it("errors on a 0-dimensional input", {
    expect_error(nv_quantile(nv_scalar(1), 0.5), "0-dimensional")
  })
})

describe("mean()", {
  it("errors when na.rm = TRUE", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_error(mean(x, na.rm = TRUE), "na.rm = TRUE")
  })

  it("errors when trim is non-zero", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_error(mean(x, trim = 0.1), "trim")
  })
})

describe("nv_argmax / nv_argmin", {
  it("default dim is the last dimension", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(nv_argmax(m), prim_argmax(m, dim = 2L))
    expect_equal(nv_argmin(m), prim_argmin(m, dim = 2L))
  })
  it("errors on a 0-dimensional input", {
    expect_error(nv_argmax(nv_scalar(3)), "0-dimensional")
    expect_error(nv_argmin(nv_scalar(3)), "0-dimensional")
  })
})

describe("cross-device eager (check_eager)", {
  # Test data
  vec_f <- nv_array(c(1, 2, 3))
  vec_f2 <- nv_array(c(4, 5, 6))
  vec_i <- nv_array(c(1L, 2L, 3L))
  vec_i2 <- nv_array(c(3L, 2L, 1L))
  vec_b <- nv_array(c(TRUE, FALSE, TRUE))
  vec_b2 <- nv_array(c(FALSE, TRUE, TRUE))
  mat_2x3 <- nv_matrix(1:6, nrow = 2)
  mat_3x3 <- nv_matrix(c(4, 2, 1, 2, 5, 3, 1, 3, 6), nrow = 3, dtype = "f32")
  sym_pd <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
  rhs_mat <- nv_matrix(c(1, 2), nrow = 2, dtype = "f32")

  it("binary arithmetic ops", {
    check_eager(nv_add, vec_f, vec_f2)
    check_eager(nv_sub, vec_f, vec_f2)
    check_eager(nv_mul, vec_f, vec_f2)
    check_eager(nv_div, vec_f, vec_f2)
    check_eager(nv_pow, vec_f, vec_f2)
    check_eager(nv_remainder, nv_array(c(7, 8, 9)), nv_array(c(3, 3, 4)))
    check_eager(nv_mod, nv_array(c(7, -7, 7)), nv_array(c(3, 3, -3)))
    check_eager(nv_max, vec_f, vec_f2)
    check_eager(nv_min, vec_f, vec_f2)
    check_eager(nv_atan2, vec_f, vec_f2)
  })

  it("binary comparison ops", {
    check_eager(nv_eq, vec_i, vec_i2)
    check_eager(nv_ne, vec_i, vec_i2)
    check_eager(nv_gt, vec_i, vec_i2)
    check_eager(nv_ge, vec_i, vec_i2)
    check_eager(nv_lt, vec_i, vec_i2)
    check_eager(nv_le, vec_i, vec_i2)
  })

  it("binary logical ops", {
    check_eager(nv_and, vec_b, vec_b2)
    check_eager(nv_or, vec_b, vec_b2)
    check_eager(nv_xor, vec_b, vec_b2)
  })

  it("binary bitwise shifts", {
    x <- nv_array(c(1L, 2L, 4L))
    y <- nv_array(c(1L, 2L, 1L))
    check_eager(nv_shift_left, x, y)
    check_eager(nv_shift_right_logical, x, y)
    check_eager(nv_shift_right_arithmetic, x, y)
  })

  it("unary math ops", {
    check_eager(nv_abs, nv_array(c(-1, 2, -3)))
    check_eager(nv_negate, vec_f)
    check_eager(nv_sqrt, nv_array(c(1, 4, 9)))
    check_eager(nv_rsqrt, nv_array(c(1, 4, 9)))
    check_eager(nv_log, nv_array(c(1, 2.7, 7.4)))
    check_eager(nv_log1p, nv_array(c(0, 0.001, 1)))
    check_eager(nv_log2, nv_array(c(1, 2, 4, 8)))
    check_eager(nv_log10, nv_array(c(1, 10, 100)))
    check_eager(nv_exp, nv_array(c(0, 1, 2)))
    check_eager(nv_expm1, nv_array(c(0, 0.001, 1)))
    check_eager(nv_cbrt, nv_array(c(1, 8, 27)))
    check_eager(nv_logistic, nv_array(c(-2, 0, 2)))
    check_eager(nv_sin, nv_array(c(0, pi / 2, pi)))
    check_eager(nv_cos, nv_array(c(0, pi / 2, pi)))
    check_eager(nv_tan, nv_array(c(0, 0.5, 1)))
    check_eager(nv_tanh, nv_array(c(-1, 0, 1)))
    check_eager(nv_floor, nv_array(c(1.2, 2.7, -1.5)))
    check_eager(nv_ceiling, nv_array(c(1.2, 2.7, -1.5)))
    check_eager(nv_trunc, nv_array(c(1.2, 2.7, -1.5)))
    check_eager(nv_sign, nv_array(c(-3, 0, 5)))
    check_eager(nv_round, nv_array(c(1.4, 2.5, 3.6)))
    check_eager(nv_popcnt, nv_array(c(7L, 3L, 15L)))
  })

  it("unary predicates", {
    x <- nv_array(c(1, NaN, Inf, -Inf, 0))
    check_eager(nv_is_finite, x)
    check_eager(nv_is_nan, x)
    check_eager(nv_is_infinite, x)
    check_eager(nv_not, vec_b)
  })

  it("conversion ops", {
    check_eager(function(x) nv_convert(x, dtype = "f64"), vec_i)
    check_eager(function(x) nv_bitcast_convert(x, dtype = "i32"), nv_array(1.5, dtype = "f32"))
  })

  it("broadcasting / shape-returning helpers", {
    check_eager(nv_broadcast_arrays, vec_f, nv_matrix(1:9, nrow = 3, ncol = 3))
    check_eager(function(x) nv_broadcast_to(x, shape = c(2, 3)), vec_f)
    check_eager(function(x) nv_broadcast_scalars(x, 1), vec_f)
    check_eager(function(...) nv_concatenate(..., dimension = 1L), vec_f, vec_f2)
    check_eager(function(...) nv_promote_to_common(...), vec_i, vec_f)
  })

  it("shape manipulation", {
    check_eager(function(x) nv_reshape(x, c(3, 2)), mat_2x3)
    check_eager(function(x) nv_transpose(x), mat_2x3)
    check_eager(function(x) nv_squeeze(x), nv_array(1:6, shape = c(1, 6, 1)))
    check_eager(function(x) nv_unsqueeze(x, dim = 1L), vec_f)
    check_eager(function(x) nv_reverse(x, dims = 1L), vec_f)
    check_eager(function(x) nv_select(x, dim = 1L, index = 1L), mat_2x3)
  })

  it("reductions", {
    check_eager(function(x) nv_reduce_sum(x, dims = 1L), mat_2x3)
    check_eager(function(x) nv_mean(x, dims = 1L), nv_matrix(1:6, nrow = 2, dtype = "f32"))
    check_eager(function(x) nv_reduce_prod(x, dims = 1L), mat_2x3)
    check_eager(function(x) nv_reduce_max(x, dims = 1L), mat_2x3)
    check_eager(function(x) nv_reduce_min(x, dims = 1L), mat_2x3)
    check_eager(function(x) nv_reduce_any(x, dims = 1L), nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
    check_eager(function(x) nv_reduce_all(x, dims = 1L), nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
    check_eager(function(x) nv_var(x, dims = 1L), nv_array(c(1, 2, 3, 4, 5), dtype = "f32"))
    check_eager(function(x) nv_sd(x, dims = 1L), nv_array(c(1, 2, 3, 4, 5), dtype = "f32"))
  })

  it("clamp / ifelse / pad", {
    check_eager(function(x) nv_clamp(0, x, 1), nv_array(c(-0.5, 0.5, 1.5)))
    check_eager(
      function(p, x, y) nv_ifelse(p, x, y),
      vec_b,
      vec_f,
      vec_f2
    )
    check_eager(
      function(x) nv_pad(x, 0, edge_padding_low = 1L, edge_padding_high = 1L),
      vec_f
    )
  })

  it("linear algebra", {
    a <- nv_matrix(1:6, nrow = 2, dtype = "f32")
    b <- nv_matrix(1:6, nrow = 3, dtype = "f32")
    sq <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    rect <- nv_matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, dtype = "f64")
    sym <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
    L <- nv_matrix(c(2, 1, 0, 3), nrow = 2, dtype = "f32")
    rhs <- nv_matrix(c(4, 3), nrow = 2, dtype = "f32")
    check_eager(nv_matmul, a, b)
    check_eager(nv_crossprod, a)
    check_eager(nv_tcrossprod, a)
    check_eager(nv_chol, sym_pd)
    check_eager(nv_solve, sq, nv_matrix(c(1, 2), nrow = 2, dtype = "f64"))
    check_eager(nv_solve, sq, nv_array(c(1, 2), dtype = "f64"))
    check_eager(nv_triangular_solve, L, rhs)
    check_eager(nv_qr, rect)
    check_eager(nv_lu, sq)
    check_eager(nv_svd, rect)
    check_eager(nv_eigh, sym)
    check_eager(nv_det, sq)
    check_eager(nv_determinant, sq)
    check_eager(function(x) nv_determinant(x, logarithm = FALSE), sq)
    check_eager(nv_inv, sq)
    check_eager(nv_diag, vec_f)
    check_eager(nv_extract_diag, mat_3x3)
    check_eager(nv_trace, mat_3x3)
    check_eager(function(x) nv_tril(x, diagonal = 0L), mat_3x3)
    check_eager(function(x) nv_triu(x, diagonal = 0L), mat_3x3)
    check_eager(nv_outer, nv_array(c(1, 2, 3)), nv_array(c(4, 5)))
  })

  it("*_like helpers forward device from `like`", {
    check_eager(function(like) nv_fill_like(like, 0, shape = c(2, 3)), vec_f)
    check_eager(function(like) nv_iota_like(like, dim = 1L, shape = 4L, dtype = "i32"), vec_f)
    check_eager(function(like) nv_eye_like(like, 3L), vec_f)
    check_eager(function(like) nv_array_like(like, c(7L, 8L, 9L)), vec_i)
    check_eager(function(like) nv_scalar_like(like, 7L), vec_i)
    check_eager(function(like) nv_empty_like(like, shape = c(2, 2)), vec_f)
  })

  it("sorting / searching", {
    sortable <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
    sortable_even <- nv_array(c(1, 2, 3, 4))
    check_eager(nv_sort, sortable)
    check_eager(nv_argsort, sortable)
    check_eager(function(x) nv_top_k(x, k = 3L), sortable)
    check_eager(nv_median, sortable)
    check_eager(nv_median, sortable_even)
    check_eager(function(x) nv_quantile(x, 0.5), sortable)
    check_eager(function(x) nv_quantile(x, array(c(0.25, 0.75))), sortable)
    check_eager(nv_argmax, sortable)
    check_eager(nv_argmin, sortable)
  })
})

# Regression for r-xla/anvl#343: R literals must adopt the device of their
# AnvlArray siblings rather than being placed on the default device.
describe("literals adopt device of array siblings", {
  dev1 <- nv_device("cpu:1", "xla")

  it("nv_ifelse with literal branches", {
    pred <- nv_array(c(TRUE, FALSE), device = dev1)
    out <- nv_ifelse(pred, 1, 2)
    expect_true(eq_device(device(out), dev1))
  })

  it("nv_ifelse with literal pred", {
    tv <- nv_array(c(1, 2), device = dev1)
    fv <- nv_array(c(3, 4), device = dev1)
    out <- nv_ifelse(arr(TRUE, FALSE), tv, fv)
    expect_true(eq_device(device(out), dev1))
  })

  it("nv_rbind with literal", {
    x <- nv_array(c(1, 2), device = dev1)
    out <- nv_rbind(x, arr(3, 4))
    expect_true(eq_device(device(out), dev1))
  })

  it("nv_cbind with literal", {
    x <- nv_array(c(1, 2), device = dev1)
    out <- nv_cbind(x, arr(3, 4))
    expect_true(eq_device(device(out), dev1))
  })
})

describe("nv_solve", {
  it("matches base R for matrix b (output stays a 2-D matrix)", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    b_mat <- matrix(c(1, 2), nrow = 2)
    x <- nv_solve(nv_array(A_mat, dtype = "f64"), nv_array(b_mat, dtype = "f64"))
    expect_equal(shape(x), c(2L, 1L))
    expect_equal(as_array(x), solve(A_mat, b_mat), tolerance = 1e-5)
  })

  it("matches base R for vector b (output stays a 1-D vector)", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    x <- nv_solve(nv_array(A_mat, dtype = "f64"), nv_array(c(1, 2), dtype = "f64"))
    expect_equal(shape(x), 2L)
    expect_equal(as_array(x), array(solve(A_mat, c(1, 2))), tolerance = 1e-5)
  })
})

describe("nv_triangular_solve", {
  it("matches base R (lower, vector b)", {
    L_mat <- matrix(c(3, 1, 0, 2), nrow = 2)
    expect_equal(
      as_array(nv_triangular_solve(
        nv_array(L_mat, dtype = "f64"),
        nv_array(c(6, 5), dtype = "f64")
      )),
      array(solve(L_mat, c(6, 5))),
      tolerance = 1e-5
    )
  })

  it("respects transpose_a = TRUE", {
    L_mat <- matrix(c(3, 1, 0, 2), nrow = 2)
    b <- c(6, 5)
    expect_equal(
      as_array(nv_triangular_solve(
        nv_array(L_mat, dtype = "f64"),
        nv_array(b, dtype = "f64"),
        transpose_a = TRUE
      )),
      array(solve(t(L_mat), b)),
      tolerance = 1e-5
    )
  })
})

describe("nv_lu", {
  it("returns L, U, pivots, permutation with the right shapes and factorization (square)", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    out <- nv_lu(nv_array(A_mat, dtype = "f64"))
    expect_named(out, c("L", "U", "pivots", "permutation"))
    expect_equal(shape(out$L), c(2L, 2L))
    expect_equal(shape(out$U), c(2L, 2L))
    expect_equal(shape(out$pivots), 2L)
    expect_equal(shape(out$permutation), 2L)
    L <- as_array(out$L)
    U <- as_array(out$U)
    permutation <- as_array(out$permutation)
    # L is unit lower-triangular, U is upper-triangular.
    expect_equal(L[upper.tri(L)], rep(0, sum(upper.tri(L))))
    expect_equal(diag(L), c(1, 1))
    expect_equal(U[lower.tri(U)], rep(0, sum(lower.tri(U))))
    # Factorization identity: P %*% A == L %*% U.
    expect_equal(L %*% U, A_mat[permutation, , drop = FALSE], tolerance = 1e-5)
  })

  it("handles a tall (m > n) matrix", {
    m <- 3L
    n <- 2L
    k <- min(m, n)
    A_mat <- matrix(c(1, 2, 3, 4, 5, 6), nrow = m)
    out <- nv_lu(nv_array(A_mat, dtype = "f64"))
    expect_equal(shape(out$L), c(m, k))
    expect_equal(shape(out$U), c(k, n))
    L <- as_array(out$L)
    U <- as_array(out$U)
    permutation <- as_array(out$permutation)
    # L: unit diagonal on the first k rows, zeros above the diagonal.
    L_top <- L[seq_len(k), , drop = FALSE]
    expect_equal(L_top[upper.tri(L_top)], rep(0, sum(upper.tri(L_top))))
    expect_equal(diag(L_top), c(1, 1))
    expect_equal(L %*% U, A_mat[permutation, , drop = FALSE], tolerance = 1e-5)
  })

  it("handles a wide (m < n) matrix", {
    skip_if(is_cuda(), "m < n not supported on CUDA yet")
    m <- 2L
    n <- 3L
    k <- min(m, n)
    A_mat <- matrix(c(1, 2, 3, 4, 5, 6), nrow = m)
    out <- nv_lu(nv_array(A_mat, dtype = "f64"))
    expect_equal(shape(out$L), c(m, k))
    expect_equal(shape(out$U), c(k, n))
    L <- as_array(out$L)
    U <- as_array(out$U)
    permutation <- as_array(out$permutation)
    expect_equal(diag(L), c(1, 1))
    expect_equal(L %*% U, A_mat[permutation, , drop = FALSE], tolerance = 1e-5)
  })
})

describe("nv_chol", {
  it("returns the upper-triangular factor matching base R", {
    A_mat <- matrix(c(4, 2, 2, 3), nrow = 2)
    U <- as_array(nv_chol(nv_array(A_mat, dtype = "f64")))
    expect_equal(t(U) %*% U, A_mat, tolerance = 1e-5)
    # `lower = TRUE` should give the corresponding lower-triangular factor.
    L <- as_array(nv_chol(nv_array(A_mat, dtype = "f64"), lower = TRUE))
    expect_equal(L %*% t(L), A_mat, tolerance = 1e-5)
  })
})

describe("nv_det", {
  it("matches base R det()", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    expect_equal(
      as_array(nv_det(nv_array(A_mat, dtype = "f64"))),
      det(A_mat),
      tolerance = 1e-5
    )
  })

  it("returns 1 for the empty 0x0 matrix", {
    empty <- nv_matrix(numeric(0), nrow = 0, ncol = 0, dtype = "f64")
    expect_equal(as_array(nv_det(empty)), 1)
  })
})

describe("nv_determinant", {
  it("logarithm = TRUE matches base::determinant()", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    out <- nv_determinant(nv_array(A_mat, dtype = "f64"), logarithm = TRUE)
    expect_equal(as_array(out$modulus), log(abs(det(A_mat))), tolerance = 1e-5)
    expect_equal(as_array(out$sign), sign(det(A_mat)), tolerance = 1e-5)
  })

  it("logarithm = FALSE returns abs(det) and the same sign", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    out <- nv_determinant(nv_array(A_mat, dtype = "f64"), logarithm = FALSE)
    expect_equal(as_array(out$modulus), abs(det(A_mat)), tolerance = 1e-5)
    expect_equal(as_array(out$sign), sign(det(A_mat)), tolerance = 1e-5)
  })

  it("handles the empty 0x0 matrix (det = 1)", {
    empty <- nv_matrix(numeric(0), nrow = 0, ncol = 0, dtype = "f64")
    out_log <- nv_determinant(empty, logarithm = TRUE)
    expect_equal(as_array(out_log$modulus), 0)
    expect_equal(as_array(out_log$sign), 1)
    out_lin <- nv_determinant(empty, logarithm = FALSE)
    expect_equal(as_array(out_lin$modulus), 1)
    expect_equal(as_array(out_lin$sign), 1)
  })
})

describe("nv_inv", {
  it("matches base R solve()", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    expect_equal(
      as_array(nv_inv(nv_array(A_mat, dtype = "f64"))),
      solve(A_mat),
      tolerance = 1e-5
    )
  })

  it("returns the empty matrix for a 0x0 input", {
    empty <- nv_matrix(numeric(0), nrow = 0, ncol = 0, dtype = "f64")
    out <- nv_inv(empty)
    expect_equal(shape(out), c(0L, 0L))
  })
})

describe("nv_flatten", {
  it("works for 2D input", {
    x <- matrix(1:4, nrow = 2)
    expect_equal(
      nv_flatten(nv_array(x)),
      nv_array(as.vector(t(x)))
    )
  })
  it("works for 1D input", {
    x <- nv_array(1:3)
    expect_equal(
      nv_flatten(x),
      x
    )
  })
  it("fails for 0D input", {
    expect_error(nv_flatten(1), "scalar")
  })
  it("works with empty input", {
    expect_equal(
      nv_flatten(nv_empty("f32", c(2, 0))),
      nv_empty("f32", 0L)
    )
  })
})

test_that("nv_mod and `%%` follow base R flooring semantics across sign combos", {
  expect_equal(
    as.vector(nv_mod(1, -3)),
    1 %% -3
  )
  expect_equal(
    as.vector(nv_mod(1.4, -0.2)),
    1.4 %% -0.2,
    tolerance = 1e-5
  )
  expect_equal(
    as.vector(nv_mod(1L, -3L)),
    1L %% -3L
  )
})
