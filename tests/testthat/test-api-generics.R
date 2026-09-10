# Base R's generics on an anvl array mean what they mean in base R, so base R is
# the reference these tests compare against. A generic that only delegates to
# its `nv_*` twin is tested with that function in test-api.R; what is tested
# here is the logic a method adds on top.

# Operators --------------------------------------------------------------------

describe("`[`", {
  # main tests are in test-api-subset.R
  it("extracts a single element", {
    x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
    expect_equal(x[1L, 1L], nv_scalar(1, dtype = "f32"))
  })

  it("can use variables as indices", {
    x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
    idx1 <- nv_scalar(2L, dtype = "i32")
    idx2 <- nv_scalar(3L, dtype = "i32")
    # Scalar array indices drop axes, so the result is a scalar.
    expect_equal(x[idx1, idx2], nv_scalar(8, dtype = "f32"))
  })

  it("says that drop is not supported", {
    x <- nv_array(matrix(1:6, 2))
    expect_error(x[1L, , drop = FALSE], "`drop` is not supported")
    expect_error(x[1L, drop = TRUE], "`drop` is not supported")
  })

  it("rejects more subset specifications than there are axes", {
    expect_error(nv_array(1:3)[1L, 1L], "Too many subset specifications")
  })
})

describe("`[<-`", {
  # main tests are in test-api-subset.R
  it("updates a single element", {
    x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
    x[3L, 4L] <- nv_scalar(-1, dtype = "f32")
    expect_equal(x, nv_array(c(1:11, -1L), dtype = "f32", shape = c(3, 4)))
  })

  it("can use variables as indices (NSE)", {
    x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
    idx1 <- nv_scalar(1L, dtype = "i32")
    idx2 <- nv_scalar(2L, dtype = "i32")
    x[idx1, idx2] <- nv_scalar(99, dtype = "f32")
    expect_equal(x, nv_array(c(1:3, 99, 5:12), dtype = "f32", shape = c(3, 4)))
  })
})

describe("`&`", {
  it("computes the logical AND of two boolean arrays", {
    p <- c(TRUE, FALSE, TRUE)
    q <- c(TRUE, TRUE, FALSE)
    out <- nv_array(p) & nv_array(q)
    expect_equal(as.vector(out), p & q)
    expect_equal(dtype(out), as_dtype("bool"))
  })

  it("rejects a non-boolean array instead of coercing it", {
    expect_error(nv_array(12L) & nv_array(10L), "must be a boolean array")
    expect_error(nv_array(1.5) & nv_array(TRUE), "must be a boolean array")
    expect_error(nv_array(TRUE) & nv_array(1.5), "must be a boolean array")
    expect_error(nv_array(12L, dtype = "ui8") & nv_array(TRUE), "must be a boolean array")
    # The explicit comparison is what base R's coercion would have done.
    vals <- c(1.5, 0, -2)
    other <- c(0, 0.5, 3)
    out <- (nv_array(vals) != 0) & (nv_array(other) != 0)
    expect_equal(as.vector(out), vals & other)
  })

  it("takes a logical R value on either side, but not a numeric one", {
    x <- nv_array(TRUE)
    expect_equal(as.vector(x & TRUE), TRUE & TRUE)
    expect_equal(as.vector(FALSE & x), FALSE & TRUE)
    expect_error(x & 2, "must be a boolean array")
    expect_error(2 & x, "must be a boolean array")
    expect_error(x & 0L, "must be a boolean array")
  })

  it("stays logical under jit()", {
    f <- function(a, b) (a & b) | !a
    args <- list(nv_array(c(TRUE, FALSE)), nv_array(c(TRUE, TRUE)))
    expect_equal(
      as.vector(do.call(jit(f), args)),
      (c(TRUE, FALSE) & c(TRUE, TRUE)) | !c(TRUE, FALSE)
    )
  })
})

describe("`|`", {
  it("computes the logical OR of two boolean arrays", {
    p <- c(TRUE, FALSE, TRUE)
    q <- c(TRUE, TRUE, FALSE)
    expect_equal(as.vector(nv_array(p) | nv_array(q)), p | q)
    # base R's xor() is built on `|` and `&`, so it follows.
    expect_equal(as.vector(xor(nv_array(p), nv_array(q))), xor(p, q))
  })

  it("rejects a non-boolean array instead of coercing it", {
    expect_error(nv_array(12L) | nv_array(10L), "must be a boolean array")
    expect_error(xor(nv_array(12L), nv_array(10L)), "must be a boolean array")
  })
})

describe("`!`", {
  it("negates a boolean array", {
    expect_equal(
      !nv_array(c(TRUE, FALSE, TRUE)),
      nv_array(c(FALSE, TRUE, FALSE))
    )
  })

  it("rejects a non-boolean array instead of coercing it", {
    expect_error(!nv_array(12L), "must be a boolean array")
    # The bitwise counterpart is still available.
    expect_equal(as.vector(nv_not(nv_array(12L))), bitwNot(12L))
  })
})

# Mathematical generics --------------------------------------------------------

describe("log", {
  it("takes a base like base R", {
    vals <- c(2, 4, 8)
    x <- nv_array(vals, dtype = "f64")
    expect_equal(as.vector(log(x, base = 2)), log(vals, base = 2), tolerance = 1e-6)
    expect_equal(as.vector(log(x, 2)), log(vals, 2), tolerance = 1e-6)
    expect_equal(as.vector(log(x)), log(vals), tolerance = 1e-6)
  })

  it("takes an array as the base", {
    x <- nv_array(c(2, 4, 8), dtype = "f64")
    expect_equal(as.vector(log(x, nv_scalar(2, "f64"))), log(c(2, 4, 8), 2), tolerance = 1e-6)
  })
})

describe("trigamma", {
  it("computes the second derivative of log-gamma, like base R", {
    vals <- c(0.5, 1, 2, 5)
    expect_equal(
      as.vector(trigamma(nv_array(vals))),
      trigamma(vals),
      tolerance = 1e-5
    )
  })
})

# Rounding ---------------------------------------------------------------------

describe("round", {
  it("takes digits like base R", {
    expect_equal(as.vector(round(nv_array(1.2345, dtype = "f64"), 2)), round(1.2345, 2))
    expect_equal(as.vector(round(nv_array(1234.5, dtype = "f64"), -2)), round(1234.5, -2))
  })

  it("rounds half to even and forwards `method`, like nv_round()", {
    expect_equal(as.vector(round(nv_array(c(0.5, 1.5)))), round(c(0.5, 1.5)))
    expect_equal(as.vector(round(nv_array(c(0.5, 1.5)), method = "afz")), c(1, 2))
  })

  it("returns an integer array unchanged", {
    x <- nv_array(3L)
    expect_equal(round(x), x)
    expect_equal(round(x, 2), x)
    expect_equal(round(nv_array(3L, dtype = "ui8")), nv_array(3L, dtype = "ui8"))
  })

  it("rejects a negative digits for an integer array", {
    expect_error(round(nv_array(15L), -1), "cannot round an integer array")
  })

  it("means the same thing under jit()", {
    f <- function(a) round(a, 1)
    x <- nv_array(c(1.25, 2.349), dtype = "f64")
    expect_equal(as.vector(jit(f)(x)), as.vector(f(x)))
  })
})

# Summary generics -------------------------------------------------------------

describe("sum", {
  it("treats unnamed extra arguments as data, like base R", {
    m <- matrix(as.double(1:6), 2)
    x <- nv_array(m)
    expect_equal(as.vector(sum(x, 2)), sum(m, 2))
    expect_equal(as.vector(sum(x, x)), sum(m, m))
  })

  it("passes named arguments to nv_reduce_sum()", {
    x <- nv_array(matrix(as.double(1:6), 2))
    expect_equal(as.vector(sum(x, axes = 1L)), c(3, 7, 11))
    expect_error(sum(nv_array(1:3), foo = "bar"), "unused argument")
  })

  it("forwards na.rm as nan_rm", {
    x <- nv_array(c(1, NaN, 3))
    expect_equal(as_array(sum(x, na.rm = TRUE)), 4)
    expect_true(is.nan(as_array(sum(x))))
  })

  it("rejects several data arguments together with a named one", {
    x <- nv_array(matrix(as.double(1:6), 2))
    expect_error(sum(x, x, axes = 1L), "cannot combine several data arguments")
  })

  it("means the same thing under jit()", {
    f <- function(a) sum(a, 2)
    x <- nv_array(matrix(as.double(1:6), 2))
    expect_equal(as.vector(jit(f)(x)), as.vector(f(x)))
  })
})

describe("prod", {
  it("treats unnamed extra arguments as data, like base R", {
    expect_equal(as.vector(prod(nv_array(c(2, 3)), 2)), prod(c(2, 3), 2))
  })

  it("passes named arguments to nv_reduce_prod()", {
    x <- nv_array(matrix(as.double(1:6), 2))
    expect_equal(as.vector(prod(x, axes = 1L)), c(2, 12, 30))
  })
})

describe("max", {
  it("treats unnamed extra arguments as data, like base R", {
    m <- matrix(as.double(1:6), 2)
    expect_equal(as.vector(max(nv_array(m), 5)), max(m, 5))
  })

  it("forwards na.rm as nan_rm", {
    x <- nv_array(c(1, NaN, 3))
    expect_equal(as_array(max(x, na.rm = TRUE)), 3)
    expect_true(is.nan(as_array(max(x))))
  })
})

describe("min", {
  it("treats unnamed extra arguments as data, like base R", {
    m <- matrix(as.double(1:6), 2)
    expect_equal(as.vector(min(nv_array(m), nv_array(-m))), min(m, -m))
  })
})

describe("range", {
  it("treats unnamed extra arguments as data, like base R", {
    m <- matrix(as.double(1:6), 2)
    expect_equal(as.vector(range(nv_array(m), 10)), range(m, 10))
    expect_equal(as.vector(range(nv_array(m), nv_array(-m))), range(m, -m))
  })

  it("passes named arguments to the underlying reductions", {
    x <- nv_array(matrix(as.double(1:6), 2))
    expect_equal(range(x, axes = 1L), nv_range(x, axes = 1L))
  })
})

describe("any", {
  it("reduces boolean arrays, like base R", {
    expect_equal(as.vector(any(nv_array(c(TRUE, FALSE)))), any(c(TRUE, FALSE)))
    expect_equal(as.vector(any(nv_array(c(TRUE, FALSE)), FALSE)), any(c(TRUE, FALSE), FALSE))
  })

  it("rejects a non-boolean argument instead of coercing it", {
    expect_error(any(nv_array(c(0L, 1L))), "must be a boolean array")
    expect_error(any(nv_array(TRUE), 1L), "must be a boolean array")
  })
})

describe("all", {
  it("reduces boolean arrays, like base R", {
    expect_equal(as.vector(all(nv_array(c(TRUE, FALSE)))), all(c(TRUE, FALSE)))
    expect_equal(as.vector(all(nv_array(c(TRUE, TRUE)), TRUE)), all(c(TRUE, TRUE), TRUE))
  })

  it("rejects a non-boolean argument instead of coercing it", {
    expect_error(all(nv_array(c(2, 3))), "must be a boolean array")
  })
})

# Array generics ---------------------------------------------------------------

describe("c", {
  it("concatenates scalars and 1-D arrays, like base R", {
    x <- nv_array(1:3)
    y <- nv_array(4:6)
    expect_equal(as.vector(c(x, y)), c(1:3, 4:6))
    expect_equal(as.vector(c(x, 4L)), c(1:3, 4L))
    expect_equal(as.vector(c(x, 4:6)), c(1:3, 4:6))
    expect_equal(as.vector(c(x)), 1:3)
    expect_equal(as.vector(c(nv_scalar(1L), nv_scalar(2L))), c(1L, 2L))
    # Promotes to a common data type, like nv_concatenate() does.
    expect_equal(as.vector(c(x, nv_array(1.5))), c(1, 2, 3, 1.5))
  })

  it("rejects an array with more than one axis", {
    m <- nv_array(matrix(1:4, 2))
    expect_error(c(m), "only scalars and 1-D arrays")
    expect_error(c(nv_array(1L), m), "only scalars and 1-D arrays")
    expect_error(c(nv_array(1L), matrix(1:4, 2)), "only scalars and 1-D arrays")
  })

  it("works under jit()", {
    f <- function(a, b) c(a, b)
    expect_equal(as.vector(jit(f)(nv_array(1:3), nv_array(4:6))), c(1:3, 4:6))
  })
})

describe("dim", {
  it("returns the shape, also for a 1-D array", {
    x <- nv_array(1:24, shape = c(2, 3, 4))
    expect_equal(dim(x), shape(x))
    expect_equal(dim(nv_array(1:3)), 3L)
  })
})

describe("length", {
  it("returns the number of elements", {
    expect_equal(length(nv_array(1:3)), 3L)
    expect_equal(length(nv_scalar(0L)), 1L)
    expect_equal(length(nv_array(1L, shape = c(2L, 3L))), 6L)
  })
})

describe("nrow", {
  it("returns the size of axis 1, also for a 1-D array", {
    expect_equal(nrow(nv_array(1:6, shape = c(3L, 2L))), 3L)
    expect_equal(nrow(nv_array(1:2)), 2L)
  })
})

describe("ncol", {
  it("returns the size of axis 2, and NA when there is none", {
    expect_equal(ncol(nv_array(1:6, shape = c(3L, 2L))), 2L)
    expect_true(is.na(ncol(nv_array(1L))))
  })
})

describe("rev", {
  it("puts the elements in base R's order", {
    expect_equal(as.vector(rev(nv_array(1:3))), rev(1:3))
    m <- matrix(1:6, nrow = 2)
    # Base R flattens; an anvl array keeps its shape, but the elements come out
    # in the same order.
    expect_equal(as.vector(rev(nv_array(m))), rev(as.vector(m)))
    expect_equal(shape(rev(nv_array(m))), shape(nv_array(m)))
  })

  it("passes a scalar array through", {
    expect_equal(as.vector(rev(nv_scalar(1L))), 1L)
  })

  it("works under jit()", {
    expect_equal(as.vector(jit(rev)(nv_array(1:3))), rev(1:3))
  })
})

describe("t", {
  it("transposes a matrix", {
    x <- nv_matrix(1:6, nrow = 2)
    expect_equal(t(x), nv_transpose(x))
  })

  it("errors on anything but a matrix, unlike base R", {
    expect_error(t(nv_scalar(1)), "requires a 2-D array")
    expect_error(t(nv_array(1:3)), "requires a 2-D array")
    expect_error(t(nv_array(array(1:24, dim = c(2, 3, 4)))), "requires a 2-D array")
  })
})

# Linear-algebra generics ------------------------------------------------------
# These check that base R's S3 dispatch reaches the corresponding nv_*
# implementation with the arguments mapped over.

describe("solve", {
  it("solves a x = b for matrix b", {
    a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    b <- nv_matrix(c(1, 2), nrow = 2, dtype = "f64")
    expect_equal(solve(a, b), nv_solve(a, b))
  })

  it("returns the inverse when b is missing", {
    a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    expect_equal(solve(a), nv_inv(a))
  })
})

describe("qr", {
  it("dispatches to nv_qr", {
    a <- nv_matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, dtype = "f64")
    expect_equal(qr(a), nv_qr(a))
  })
})

describe("chol", {
  it("returns the upper-triangular factor (base R convention)", {
    a <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f64")
    expect_equal(as_array(chol(a)), as_array(nv_chol(a)), tolerance = 1e-5)
  })

  it("respects lower = TRUE", {
    a <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f64")
    expect_equal(chol(a, lower = TRUE), nv_chol(a, lower = TRUE))
  })
})

describe("determinant", {
  it("takes the logarithm by default, like base R", {
    a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    expect_equal(determinant(a), nv_determinant(a, logarithm = TRUE))
  })

  it("respects logarithm = FALSE", {
    a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    expect_equal(determinant(a, logarithm = FALSE), nv_determinant(a, logarithm = FALSE))
  })
})
