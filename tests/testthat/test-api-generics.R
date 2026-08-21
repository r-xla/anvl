describe("[", {
  # main tests are in test-api-subset.R
  it("extracts single element", {
    expect_equal(
      {
        x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
        idx1 <- nv_scalar(1L, dtype = "i32")
        idx2 <- nv_scalar(1L, dtype = "i32")
        x[idx1, idx2]
      },
      nv_scalar(1)
    )
  })

  it("can use variables as indices", {
    expect_equal(
      {
        x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
        idx1 <- nv_scalar(2L, dtype = "i32")
        idx2 <- nv_scalar(3L, dtype = "i32")
        x[idx1, idx2]
      },
      # Scalar array indices drop axes, so result is a scalar
      nv_scalar(8)
    )
  })
})

describe("!", {
  it("negates boolean array", {
    expect_equal(
      {
        x <- nv_array(c(TRUE, FALSE, TRUE), dtype = "bool")
        !x
      },
      nv_array(c(FALSE, TRUE, FALSE), dtype = "bool")
    )
  })
})

describe("t", {
  it("errors on non-2D arrays", {
    expect_error(t(nv_scalar(1)), "requires a 2-D array")
    expect_error(t(nv_array(1:3)), "requires a 2-D array")
    expect_error(t(nv_array(array(1:24, dim = c(2, 3, 4)))), "requires a 2-D array")
  })
})

describe("trunc", {
  it("rounds toward zero", {
    expect_equal(
      {
        x <- nv_array(c(1.2, 2.7, -1.5, -0.3, 0))
        trunc(x)
      },
      nv_array(c(1, 2, -1, 0, 0))
    )
  })
})

describe("Math group generic rejects unsupported args", {
  it("log(x, base) errors (base is not supported)", {
    x <- nv_array(c(2, 4, 8))
    expect_error(log(x, base = 2), "unused argument")
  })
  it("sqrt(x, foo) errors", {
    expect_error(sqrt(nv_array(c(1, 4)), foo = "bar"), "sqrt")
  })
})

describe("log2", {
  it("computes base-2 logarithm", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 4, 8))
        log2(x)
      },
      nv_array(log2(c(1, 2, 4, 8))),
      tolerance = 1e-6
    )
  })
})

describe("log10", {
  it("computes base-10 logarithm", {
    expect_equal(
      {
        x <- nv_array(c(1, 10, 100, 1000))
        log10(x)
      },
      nv_array(log10(c(1, 10, 100, 1000))),
      tolerance = 1e-6
    )
  })
})

describe("expm1", {
  it("computes exp(x) - 1", {
    expect_equal(
      {
        x <- nv_array(c(0, 0.001, 1))
        expm1(x)
      },
      nv_array(expm1(c(0, 0.001, 1))),
      tolerance = 1e-6
    )
  })
})

describe("log1p", {
  it("computes log(1 + x)", {
    expect_equal(
      {
        x <- nv_array(c(0, 0.001, 1))
        log1p(x)
      },
      nv_array(log1p(c(0, 0.001, 1))),
      tolerance = 1e-6
    )
  })
})

describe("CHLO Math generics", {
  it("acos / asin / atan dispatch through Math", {
    vals <- c(-0.5, 0, 0.5)
    expect_equal(
      {
        x <- nv_array(vals)
        acos(x)
      },
      nv_array(acos(vals)),
      tolerance = 1e-6
    )
    expect_equal(
      {
        x <- nv_array(vals)
        asin(x)
      },
      nv_array(asin(vals)),
      tolerance = 1e-6
    )
    expect_equal(
      {
        x <- nv_array(vals)
        atan(x)
      },
      nv_array(atan(vals)),
      tolerance = 1e-6
    )
  })

  it("acosh / asinh / atanh dispatch through Math", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3))
        acosh(x)
      },
      nv_array(acosh(c(1, 2, 3))),
      tolerance = 1e-6
    )
    expect_equal(
      {
        x <- nv_array(c(-1, 0, 1))
        asinh(x)
      },
      nv_array(asinh(c(-1, 0, 1))),
      tolerance = 1e-6
    )
    expect_equal(
      {
        x <- nv_array(c(-0.5, 0, 0.5))
        atanh(x)
      },
      nv_array(atanh(c(-0.5, 0, 0.5))),
      tolerance = 1e-6
    )
  })

  it("cosh / sinh dispatch through Math", {
    vals <- c(-1, 0, 1)
    expect_equal(
      {
        x <- nv_array(vals)
        cosh(x)
      },
      nv_array(cosh(vals)),
      tolerance = 1e-6
    )
    expect_equal(
      {
        x <- nv_array(vals)
        sinh(x)
      },
      nv_array(sinh(vals)),
      tolerance = 1e-6
    )
  })

  it("digamma / lgamma / trigamma dispatch through Math", {
    vals <- c(0.5, 1, 2, 5)
    expect_equal(
      {
        x <- nv_array(vals)
        digamma(x)
      },
      nv_array(digamma(vals)),
      tolerance = 1e-5
    )
    expect_equal(
      {
        x <- nv_array(vals)
        lgamma(x)
      },
      nv_array(lgamma(vals)),
      tolerance = 1e-5
    )
    expect_equal(
      {
        x <- nv_array(vals)
        trigamma(x)
      },
      nv_array(trigamma(vals)),
      tolerance = 1e-5
    )
  })

  it("nv_polygamma broadcasts a scalar n", {
    vals <- c(0.5, 1, 2, 5)
    expect_equal(
      {
        x <- nv_array(vals)
        nv_polygamma(2, x)
      },
      nv_array(psigamma(vals, 2)),
      tolerance = 1e-5
    )
  })
})

describe("range", {
  it("returns c(min, max) as a length-2 array", {
    expect_equal(
      {
        x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
        range(x)
      },
      nv_array(c(1, 9))
    )
  })
})

describe("Summary group generic", {
  it("forwards ... to underlying nv_reduce_*", {
    expect_equal(
      {
        x <- nv_array(matrix(1:6, 2))
        sum(x, axes = 1L)
      },
      nv_array(c(3, 7, 11), dtype = "i32")
    )
  })
  it("forwards na.rm to the underlying nv_reduce_*", {
    # Float input with NaN: na.rm = TRUE skips it.
    x <- nv_array(c(1, NaN, 3))
    expect_equal(as_array(sum(x, na.rm = TRUE)), 4)
    expect_equal(as_array(max(x, na.rm = TRUE)), 3)
    # Without na.rm, NaN propagates.
    expect_true(is.nan(as_array(sum(x))))
    expect_true(is.nan(as_array(max(x))))
  })
  it("rejects unsupported args", {
    expect_error(sum(nv_array(1:3), foo = "bar"), "unused argument")
  })
})

describe("[<-", {
  # main tests are in test-api-subset.R
  it("updates single element", {
    expect_equal(
      {
        x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
        idx1 <- nv_scalar(3L, dtype = "i32")
        idx2 <- nv_scalar(4L, dtype = "i32")
        value <- nv_scalar(-1, dtype = "f32")
        x[3L, 4L] <- value
        x
      },
      nv_array(c(1:11, -1L), dtype = "f32", shape = c(3, 4))
    )
  })

  it("can use variables as indices (NSE)", {
    expect_equal(
      {
        x <- nv_array(1:12, dtype = "f32", shape = c(3, 4))
        idx1 <- nv_scalar(1L, dtype = "i32")
        idx2 <- nv_scalar(2L, dtype = "i32")
        value <- nv_scalar(99, dtype = "f32")
        x[idx1, idx2] <- value
        x
      },
      nv_array(c(1:3, 99, 5:12), dtype = "f32", shape = c(3, 4))
    )
  })
})

describe("length", {
  it("works with 1D array", {
    expect_equal(length(nv_array(1:3)), 3L)
  })
  it("works with scalar", {
    expect_equal(length(nv_scalar(0L)), 1L)
  })
  it("works with 2D array", {
    expect_equal(length(nv_array(1L, shape = c(2L, 3L))), 6L)
  })
})

describe("nrow", {
  it("works with >= 2D array", {
    expect_equal(nrow(nv_array(1:6, shape = c(3L, 2L))), 3L)
  })
  it("returns NULL for < 2D array", {
    expect_equal(nrow(nv_array(1:2)), 2L)
  })
})

describe("ncol", {
  it("works with >= 2D array", {
    expect_equal(ncol(nv_array(1:6, shape = c(3L, 2L))), 2L)
  })
  it("returns NULL for < 2D array", {
    expect_true(is.na(ncol(nv_array(1L))))
  })
})

describe("axis", {
  it("returns shape", {
    x <- nv_array(1:24, shape = c(2, 3, 4))
    expect_equal(dim(x), shape(x))
  })
})

describe("as.double", {
  it("returns a bare double vector and discards shape", {
    x <- nv_array(c(1, 2, 3, 4, 5, 6), dtype = "f32", shape = c(2L, 3L))
    result <- as.double(x)
    expect_type(result, "double")
    expect_null(dim(result))
    expect_equal(result, c(1, 2, 3, 4, 5, 6))
  })

  it("is equivalent to as.numeric", {
    x <- nv_array(c(1.5, 2.5), dtype = "f64")
    expect_identical(as.numeric(x), as.double(x))
  })

  it("works on a scalar", {
    expect_identical(as.double(nv_scalar(3.5, dtype = "f64")), 3.5)
  })

  it("works on signed integer dtypes", {
    x <- nv_array(1:4, dtype = "i32", shape = c(2L, 2L))
    result <- as.double(x)
    expect_type(result, "double")
    expect_null(dim(result))
    expect_equal(result, c(1, 2, 3, 4))
  })

  it("works on unsigned integer dtypes", {
    expect_identical(as.double(nv_array(c(1L, 2L, 3L), dtype = "ui32")), c(1, 2, 3))
  })

  it("errors on bool dtype", {
    expect_error(as.double(nv_array(TRUE, dtype = "bool")), "requires a float or integer dtype")
  })
})

describe("as.integer", {
  it("returns a bare integer vector and discards shape", {
    x <- nv_array(1:4, dtype = "i32", shape = c(2L, 2L))
    result <- as.integer(x)
    expect_type(result, "integer")
    expect_null(dim(result))
    expect_equal(result, 1:4)
  })

  it("works on unsigned integer dtypes", {
    x <- nv_array(c(1L, 2L, 3L), dtype = "ui32")
    expect_identical(as.integer(x), c(1L, 2L, 3L))
  })

  it("works on a scalar", {
    expect_identical(as.integer(nv_scalar(7L, dtype = "i32")), 7L)
  })

  it("errors on non-integer dtype", {
    expect_error(as.integer(nv_array(1.5, dtype = "f32")), "requires a .* integer dtype")
    expect_error(as.integer(nv_array(TRUE, dtype = "bool")), "requires a .* integer dtype")
  })
})

# Linear-algebra S3 generics (added when the linalg primitives landed).
# These just check that base R's S3 dispatch reaches the corresponding
# nv_* implementations on AnvlArray inputs.

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
    expect_equal(
      chol(a, lower = TRUE),
      nv_chol(a, lower = TRUE)
    )
  })
})

describe("determinant", {
  it("dispatches to nv_determinant (logarithm = TRUE by default)", {
    a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    out <- determinant(a)
    expected <- nv_determinant(a, logarithm = TRUE)
    expect_equal(out, expected)
  })

  it("respects logarithm = FALSE", {
    a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
    out <- determinant(a, logarithm = FALSE)
    expected <- nv_determinant(a, logarithm = FALSE)
    expect_equal(out, expected)
  })
})

describe("as.logical", {
  it("returns a bare logical vector and discards shape", {
    x <- nv_array(c(TRUE, FALSE, TRUE, FALSE), dtype = "bool", shape = c(2L, 2L))
    result <- as.logical(x)
    expect_type(result, "logical")
    expect_null(dim(result))
    expect_equal(result, c(TRUE, FALSE, TRUE, FALSE))
  })

  it("errors on non-bool dtype", {
    expect_error(as.logical(nv_array(1L, dtype = "i32")), "requires a .*bool.* dtype")
    expect_error(as.logical(nv_array(1.5, dtype = "f32")), "requires a .*bool.* dtype")
  })
})
