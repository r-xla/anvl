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
      nv_scalar(1, dtype = "f32")
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
      nv_scalar(8, dtype = "f32")
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
      nv_array(c(3, 7, 11), dtype = default_int())
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

# The operators and group generics are supposed to mean what they mean in base
# R, so base R is the reference these tests compare against.

describe("logical operators", {
  it("are logical for a non-boolean array, like base R", {
    x <- nv_array(12L)
    y <- nv_array(10L)
    expect_equal(as.vector(x & y), 12L & 10L)
    expect_equal(as.vector(x | y), 12L | 10L)
    expect_equal(as.vector(!x), !12L)
    expect_equal(as.vector(xor(x, y)), xor(12L, 10L))
    expect_equal(dtype(x & y), as_dtype("bool"))
  })

  it("compare a float array against zero, like base R", {
    vals <- c(1.5, 0, -2)
    other <- c(0, 0.5, 3)
    x <- nv_array(vals)
    y <- nv_array(other)
    expect_equal(as.vector(x & y), vals & other)
    expect_equal(as.vector(x | y), vals | other)
    expect_equal(as.vector(!x), !vals)
  })

  it("leave a boolean array alone", {
    p <- c(TRUE, FALSE, TRUE)
    q <- c(TRUE, TRUE, FALSE)
    x <- nv_array(p)
    y <- nv_array(q)
    expect_equal(as.vector(x & y), p & q)
    expect_equal(as.vector(x | y), p | q)
    expect_equal(as.vector(!x), !p)
    expect_equal(as.vector(xor(x, y)), xor(p, q))
  })

  it("coerce an R value on either side, like base R", {
    x <- nv_array(TRUE)
    expect_equal(as.vector(x & 2), TRUE & 2)
    expect_equal(as.vector(2 & x), 2 & TRUE)
    expect_equal(as.vector(x & 0L), TRUE & 0L)
    expect_equal(as.vector(x | FALSE), TRUE | FALSE)
  })

  it("also work for unsigned integers", {
    x <- nv_array(12L, dtype = "ui8")
    expect_equal(as.vector(x & nv_array(0L, dtype = "ui8")), FALSE)
    expect_equal(as.vector(!x), FALSE)
  })

  it("reject an operand that is neither arrayish nor numeric", {
    expect_error(nv_array(TRUE) & "a", "numeric or logical")
  })

  it("keep nv_and() / nv_or() / nv_xor() / nv_not() bitwise", {
    x <- nv_array(12L)
    y <- nv_array(10L)
    expect_equal(as.vector(nv_and(x, y)), bitwAnd(12L, 10L))
    expect_equal(as.vector(nv_or(x, y)), bitwOr(12L, 10L))
    expect_equal(as.vector(nv_xor(x, y)), bitwXor(12L, 10L))
    expect_equal(as.vector(nv_not(x)), bitwNot(12L))
  })

  it("mean the same thing under jit()", {
    f <- function(a, b) (a & b) | !a
    args <- list(nv_array(c(3L, 0L)), nv_array(c(1L, 1L)))
    expect_equal(as.vector(do.call(f, args)), as.vector(do.call(jit(f), args)))
    expect_equal(as.vector(do.call(f, args)), (c(3L, 0L) & c(1L, 1L)) | !c(3L, 0L))
  })
})

describe("%/%", {
  it("floors like base R, at both signs and both categories", {
    for (lhs in c(7L, -7L)) {
      for (rhs in c(2L, -2L)) {
        expect_equal(
          as.vector(nv_array(lhs) %/% nv_array(rhs)),
          lhs %/% rhs,
          info = sprintf("%d %%/%% %d", lhs, rhs)
        )
        expect_equal(
          as.vector(nv_array(as.double(lhs)) %/% nv_array(as.double(rhs))),
          as.double(lhs) %/% as.double(rhs),
          info = sprintf("%g %%/%% %g", lhs, rhs)
        )
      }
    }
    expect_equal(as.vector(nv_array(7.5) %/% nv_array(2.5)), 7.5 %/% 2.5)
  })

  it("works for unsigned integers", {
    expect_equal(
      as.vector(nv_array(7L, dtype = "ui8") %/% nv_array(2L, dtype = "ui8")),
      7L %/% 2L
    )
  })

  it("agrees with nv_mod(), i.e. (x %/% y) * y + x %% y == x", {
    x <- nv_array(c(7L, -7L, 8L, -8L))
    y <- nv_array(c(3L, 3L, -3L, -3L))
    expect_equal(as.vector(nv_int_div(x, y) * y + nv_mod(x, y)), as.vector(x))
  })
})

describe("Math group generic completeness", {
  it("computes gamma() like base R", {
    vals <- c(-2.5, -0.5, 0.5, 1, 2, 5)
    expect_equal(
      as.vector(gamma(nv_array(vals, dtype = "f64"))),
      gamma(vals),
      tolerance = 1e-6
    )
    # The poles at the non-positive integers.
    expect_true(all(is.nan(as.vector(gamma(nv_array(c(0, -1, -2)))))))
  })

  it("computes sinpi() / cospi() / tanpi() like base R", {
    vals <- c(-2.5, -0.5, 0.25, 0.5, 1, 1.5, 2, 3.7)
    x <- nv_array(vals, dtype = "f64")
    expect_equal(as.vector(sinpi(x)), sinpi(vals), tolerance = 1e-6)
    expect_equal(as.vector(cospi(x)), cospi(vals), tolerance = 1e-6)
    expect_equal(as.vector(tanpi(x)), suppressWarnings(tanpi(vals)), tolerance = 1e-6)
    # Exact at the (half-)integers, like base R.
    expect_identical(as.vector(sinpi(nv_array(c(0, 1, 2, -3)))), c(0, 0, 0, 0))
    expect_identical(as.vector(cospi(nv_array(c(0.5, 1.5, -0.5)))), c(0, 0, 0))
  })

  it("computes signif() like base R", {
    vals <- c(1.2345, 123450, -0.00012345, 0, Inf, NaN)
    x <- nv_array(vals, dtype = "f64")
    expect_equal(as.vector(signif(x, 3)), signif(vals, 3), tolerance = 1e-6)
    expect_equal(as.vector(signif(x)), signif(vals), tolerance = 1e-6)
  })

  it("takes round()'s digits like base R", {
    expect_equal(as.vector(round(nv_array(1.2345, dtype = "f64"), 2)), round(1.2345, 2))
    expect_equal(as.vector(round(nv_array(1234.5, dtype = "f64"), -2)), round(1234.5, -2))
    # Half-to-even, as base R does, and `method` still works.
    expect_equal(as.vector(round(nv_array(c(0.5, 1.5)))), round(c(0.5, 1.5)))
    expect_equal(as.vector(round(nv_array(c(0.5, 1.5)), method = "afz")), c(1, 2))
  })

  it("takes log()'s base like base R", {
    x <- nv_array(c(2, 4, 8), dtype = "f64")
    expect_equal(as.vector(log(x, base = 2)), log(c(2, 4, 8), base = 2), tolerance = 1e-6)
    expect_equal(as.vector(log(x, 2)), log(c(2, 4, 8), 2), tolerance = 1e-6)
    expect_equal(as.vector(log(x)), log(c(2, 4, 8)), tolerance = 1e-6)
  })

  it("leaves an integer array alone when rounding, like base R", {
    x <- nv_array(3L)
    expect_equal(as.vector(round(x)), round(3L))
    expect_equal(as.vector(round(x, 2)), round(3L, 2))
    expect_equal(as.vector(floor(x)), floor(3L))
    expect_equal(as.vector(ceiling(x)), ceiling(3L))
    expect_equal(as.vector(trunc(x)), trunc(3L))
    expect_equal(dtype(floor(x)), dtype(x))
  })

  it("asks for a float array where the operation needs one", {
    expect_error(signif(nv_array(3L), 2), "requires a float array")
    expect_error(round(nv_array(3L), -2), "requires a float array")
  })

  it("means the same thing under jit()", {
    f <- function(a) gamma(a) + sinpi(a) + signif(a, 2) + round(a, 1) + log(a, 2)
    args <- list(nv_array(c(1.5, 2.25), dtype = "f64"))
    expect_equal(as.vector(do.call(f, args)), as.vector(do.call(jit(f), args)))
  })
})

describe("Summary group generic data arguments", {
  it("treats unnamed extra arguments as data, like base R", {
    m <- matrix(as.double(1:6), 2)
    x <- nv_array(m)
    expect_equal(as.vector(sum(x, 2)), sum(m, 2))
    expect_equal(as.vector(sum(x, x)), sum(m, m))
    expect_equal(as.vector(prod(nv_array(c(2, 3)), 2)), prod(c(2, 3), 2))
    expect_equal(as.vector(max(x, 5)), max(m, 5))
    expect_equal(as.vector(min(x, nv_array(-m))), min(m, -m))
    expect_equal(as.vector(range(x, 10)), range(m, 10))
  })

  it("still reduces a single axis when `axes` is named", {
    x <- nv_array(matrix(as.double(1:6), 2))
    expect_equal(as.vector(sum(x, axes = 1L)), c(3, 7, 11))
  })

  it("rejects several data arguments together with `axes`", {
    x <- nv_array(matrix(as.double(1:6), 2))
    expect_error(sum(x, x, axes = 1L), "cannot combine several data arguments")
  })

  it("makes any() / all() logical, like base R", {
    expect_equal(as.vector(any(nv_array(c(0L, 1L)))), any(c(0L, 1L)))
    expect_equal(as.vector(any(nv_array(c(0L, 0L)))), any(c(0L, 0L)))
    # base R warns when it coerces a double; anvl does not.
    expect_equal(as.vector(all(nv_array(c(2, 3)))), suppressWarnings(all(c(2, 3))))
    expect_equal(as.vector(all(nv_array(c(2, 0)))), suppressWarnings(all(c(2, 0))))
    expect_equal(as.vector(any(nv_array(c(TRUE, FALSE)), FALSE)), any(c(TRUE, FALSE), FALSE))
    expect_equal(as.vector(all(nv_array(c(TRUE, TRUE)), TRUE)), all(c(TRUE, TRUE), TRUE))
    # The named functions keep requiring a boolean array.
    expect_error(nv_reduce_any(nv_array(1L)), "boolean data type")
  })

  it("means the same thing under jit()", {
    f <- function(a) sum(a, 2)
    args <- list(nv_array(matrix(as.double(1:6), 2)))
    expect_equal(as.vector(do.call(f, args)), as.vector(do.call(jit(f), args)))
  })
})

describe("median at a non-float data type", {
  it("computes at the default float, like base R", {
    expect_equal(as.vector(median(nv_array(1:4))), median(1:4))
    expect_equal(as.vector(median(nv_array(c(1L, 3L, 2L)))), median(c(1L, 3L, 2L)))
    expect_equal(as.vector(median(nv_array(c(TRUE, FALSE)))), median(c(TRUE, FALSE)))
    expect_equal(dtype(median(nv_array(1:4))), default_float())
    expect_equal(as.vector(nv_quantile(nv_array(1:4), 0.25)), 1.75)
  })

  it("honours the default data types", {
    with_default_dtypes(c(float = "f64", int = "i64"), {
      out <- median(nv_array(1:4))
      expect_equal(dtype(out), as_dtype("f64"))
      expect_equal(as.vector(out), median(1:4))
    })
  })

  it("keeps a float array's data type", {
    x <- nv_array(c(1, 2, 3, 4), dtype = "f64")
    expect_equal(dtype(median(x)), as_dtype("f64"))
    expect_equal(as.vector(median(x)), median(c(1, 2, 3, 4)))
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

describe("c", {
  it("flattens and concatenates, like base R", {
    x <- nv_array(1:3)
    y <- nv_array(4:6)
    expect_equal(as.vector(c(x, y)), c(1:3, 4:6))
    expect_equal(as.vector(c(x, 4L)), c(1:3, 4L))
    expect_equal(as.vector(c(x, 4:6)), c(1:3, 4:6))
    expect_equal(as.vector(c(x)), 1:3)
    # Promotes to a common data type, like nv_concatenate() does.
    expect_equal(as.vector(c(x, nv_array(1.5))), c(1, 2, 3, 1.5))
  })

  it("flattens an anvl array in row-major order", {
    # The documented difference to base R, which flattens column-major.
    m <- nv_array(matrix(1:4, 2))
    expect_equal(as.vector(c(m, nv_array(5L))), c(as.vector(nv_flatten(m)), 5L))
    expect_equal(as.vector(c(m, nv_array(5L))), c(1L, 3L, 2L, 4L, 5L))
    # An R matrix argument keeps base R's column-major order.
    expect_equal(as.vector(c(nv_array(0L), matrix(1:4, 2))), c(0L, 1:4))
  })

  it("works under jit()", {
    f <- function(a, b) c(a, b)
    expect_equal(as.vector(jit(f)(nv_array(1:3), nv_array(4:6))), c(1:3, 4:6))
  })
})

describe("[ with drop", {
  it("says that drop is not supported", {
    x <- nv_array(matrix(1:6, 2))
    expect_error(x[1L, , drop = FALSE], "`drop` is not supported")
    expect_error(x[1L, drop = TRUE], "`drop` is not supported")
  })
})
