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

describe("nv_broadcast_to()", {
  it("aligns the array's axes with the leading axes of the target", {
    x <- nv_array(c(1, 2))
    got <- nv_broadcast_to(x, shape = c(2L, 3L))
    expect_shape(got, c(2L, 3L))
    # every column is `x`
    expect_equal(as_array(got), matrix(c(1, 2), nrow = 2L, ncol = 3L))
  })

  it("refuses a vector that would only fit the trailing axis", {
    expect_error(nv_broadcast_to(nv_array(c(1, 2, 3)), shape = c(2L, 3L)), "3")
  })

  it("expands a size-1 axis wherever it sits", {
    x <- nv_array(c(1, 2, 3), shape = c(1L, 3L))
    expect_equal(
      as_array(nv_broadcast_to(x, shape = c(2L, 3L))),
      matrix(c(1, 2, 3), nrow = 2L, ncol = 3L, byrow = TRUE)
    )
  })
})

describe("nv_broadcast_arrays()", {
  it("appends size-1 axes to the shorter shape", {
    m <- nv_matrix(1:6, nrow = 2L)
    v <- nv_array(c(10L, 20L))
    xs <- nv_broadcast_arrays(m, v)
    expect_shape(xs[[1L]], c(2L, 3L))
    expect_shape(xs[[2L]], c(2L, 3L))
    # base R's flat recycling happens to agree when the vector is as long
    # as the first axis, which is the case here
    expect_equal(as_array(xs[[1L]] + xs[[2L]]), matrix(1:6, nrow = 2L) + c(10L, 20L))
  })

  it("refuses shapes that meet at an axis where neither size is 1", {
    expect_error(
      nv_broadcast_arrays(nv_matrix(1:6, nrow = 2L), nv_array(c(10, 20, 30))),
      "not broadcastable"
    )
  })
})

test_that("broadcasting scalars", {
  # An empty `...` used to fail inside `hlo_return()` instead of saying what
  # was missing.
  expect_error(nv_broadcast_scalars(), "At least one array is required")
  expect_error(nv_broadcast_arrays(), "At least one array is required")
  expect_error(nv_promote_to_common(), "At least one array is required")
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

test_that("nv_clamp converts min and max to the input dtype", {
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
  it("needs at least one array", {
    # An empty `...` never reaches stablehlo, so nothing downstream caught it:
    # this used to reach `max()` and return `-Inf` with an R warning.
    expect_error(nv_concatenate(), "At least one array is required")
  })
  it("auto-promotes to common", {
    expect_equal(
      nv_concatenate(nv_array(c(1, 2)), nv_array(3:4)),
      nv_array(c(1, 2, 3, 4))
    )
  })
  it("can concatenate literals", {
    expect_equal(
      nv_concatenate(1L, 2L),
      nv_array(1:2)
    )
    expect_equal(
      nv_concatenate(1L, 2L, axis = 1L),
      nv_array(1:2)
    )
    expect_equal(
      nv_concatenate(nv_array(1:2), 3L),
      nv_array(1:3)
    )
    expect_equal(
      nv_concatenate(nv_array(1L), 2L),
      nv_array(1:2)
    )
  })
  it("fails when axis is out of bounds", {
    expect_error(
      nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), axis = 3L)
    )
  })
  it("accepts a negative dimension", {
    x <- nv_array(1:6, shape = c(2, 3))
    expect_equal(nv_concatenate(x, x, axis = -1L), nv_concatenate(x, x, axis = 2L))
    expect_error(
      nv_concatenate(x, x, axis = -3L),
      "between 1 and 2, or between -2 and -1"
    )
  })
  it("can concatenate 2d arrays", {
    expect_equal(
      nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), axis = 1L),
      nv_array(1:4, shape = c(4, 1))
    )
    expect_equal(
      nv_concatenate(nv_array(1:2, shape = c(2, 1)), nv_array(3:4, shape = c(2, 1)), axis = 2L),
      nv_array(1:4, shape = c(2, 2), dtype = default_int())
    )
  })
  it("fails with incompatible shapes", {
    expect_error(
      nv_concatenate(nv_array(1, shape = c(1, 1, 1)), nv_array(2, shape = c(1, 1)), axis = 1L)
    )
  })
})

describe("nv_rbind", {
  it("needs at least one array", {
    expect_error(nv_rbind(), "At least one array is required")
  })
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

  it("treats a 1-D input as a row when mixed with a matrix", {
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

  it("promotes inputs to a common dtype", {
    x <- nv_array(c(1L, 2L, 3L))
    y <- nv_array(c(4, 5, 6))
    out <- rbind(x, y)
    expect_dtype(out, dtype(y))
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

  it("stacks two 3-D arrays along axis 1", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    b <- array(101:124, dim = c(2L, 3L, 4L))
    out <- rbind(nv_array(a), nv_array(b))
    expect_shape(out, c(4L, 3L, 4L))
    expect_equal(as_array(out)[1:2, , ], a)
    expect_equal(as_array(out)[3:4, , ], b)
  })

  it("broadcasts a scalar against a 3-D array", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    out <- rbind(nv_array(a), nv_scalar(0))
    expect_shape(out, c(3L, 3L, 4L))
    expect_equal(as_array(out)[1:2, , ], a)
    expect_equal(as_array(out)[3L, , ], array(0, dim = c(3L, 4L)))
  })
})

describe("nv_cbind", {
  it("needs at least one array", {
    expect_error(nv_cbind(), "At least one array is required")
  })
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

  it("treats a 1-D input as a column when mixed with a matrix", {
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

  it("stacks two 3-D arrays along axis 2", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    b <- array(101:124, dim = c(2L, 3L, 4L))
    out <- cbind(nv_array(a), nv_array(b))
    expect_shape(out, c(2L, 6L, 4L))
    expect_equal(as_array(out)[, 1:3, ], a)
    expect_equal(as_array(out)[, 4:6, ], b)
  })

  it("broadcasts a scalar against a 3-D array", {
    a <- array(1:24, dim = c(2L, 3L, 4L))
    out <- cbind(nv_array(a), nv_scalar(0))
    expect_shape(out, c(2L, 4L, 4L))
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

describe("nv_gamma", {
  it("computes the gamma function like base R", {
    vals <- c(-2.5, -0.5, 0.5, 1, 2, 5)
    expect_equal(
      as.vector(nv_gamma(nv_array(vals, dtype = "f64"))),
      gamma(vals),
      tolerance = 1e-6
    )
  })

  it("is NaN at the poles, like base R", {
    expect_true(all(is.nan(as.vector(nv_gamma(nv_array(c(0, -1, -2)))))))
  })

  it("computes an integer array at the default float", {
    out <- nv_gamma(nv_array(1:5))
    expect_dtype(out, default_float())
    expect_equal(as.vector(out), gamma(1:5), tolerance = 1e-5)
  })

  it("has the gradient gamma(x) * digamma(x), also at a whole number", {
    # The reflection branch is 0 * Inf at a positive whole number, which used
    # to reach the gradient through the cotangent of the discarded branch.
    vals <- c(-2.25, -1.5, -0.5, 0.5, 1, 2, 4, 7.25, 20)
    f <- function(x) nv_reduce_sum(nv_gamma(x))
    expect_equal(
      as.vector(jit(gradient(f, wrt = "x"))(nv_array(vals, dtype = "f64"))[[1L]]),
      gamma(vals) * digamma(vals),
      tolerance = 1e-6
    )
  })
})

describe("nv_sinpi", {
  it("computes sin(pi * x) like base R", {
    vals <- c(-2.5, -0.5, 0.25, 0.5, 1, 1.5, 2, 3.7)
    expect_equal(
      as.vector(nv_sinpi(nv_array(vals, dtype = "f64"))),
      sinpi(vals),
      tolerance = 1e-6
    )
  })

  it("is exact at the whole numbers, like base R", {
    expect_identical(as.vector(nv_sinpi(nv_array(c(0, 1, 2, -3)))), c(0, 0, 0, 0))
  })

  it("computes an integer array at the default float", {
    out <- nv_sinpi(nv_array(c(0L, 1L, 2L)))
    expect_dtype(out, default_float())
    expect_identical(as.vector(out), c(0, 0, 0))
  })
})

describe("nv_cospi", {
  it("computes cos(pi * x) like base R", {
    vals <- c(-2.5, -0.5, 0.25, 0.5, 1, 1.5, 2, 3.7)
    expect_equal(
      as.vector(nv_cospi(nv_array(vals, dtype = "f64"))),
      cospi(vals),
      tolerance = 1e-6
    )
  })

  it("is exact at the half integers, like base R", {
    expect_identical(as.vector(nv_cospi(nv_array(c(0.5, 1.5, -0.5)))), c(0, 0, 0))
  })
})

describe("nv_tanpi", {
  it("computes tan(pi * x) like base R", {
    vals <- c(-2.5, -0.5, 0.25, 0.5, 1, 1.5, 2, 3.7)
    expect_equal(
      as.vector(nv_tanpi(nv_array(vals, dtype = "f64"))),
      suppressWarnings(tanpi(vals)),
      tolerance = 1e-6
    )
  })

  it("is NaN at the poles, like base R", {
    expect_true(all(is.nan(as.vector(nv_tanpi(nv_array(c(0.5, -0.5, 1.5)))))))
  })
})

describe("make_float_unary", {
  # Every unary `nv_*` function that computes in floating point is built by
  # this factory, so one representative is enough to cover all of them.
  it("computes an int-like array at the default float, like base R", {
    out <- nv_sqrt(nv_array(1L))
    expect_dtype(out, default_float())
    expect_equal(as.vector(out), 1)
    expect_equal(as.vector(nv_sqrt(nv_array(4L, dtype = "ui8"))), 2)
    expect_dtype(nv_sqrt(nv_array(4L, dtype = "i64")), default_float())
    expect_equal(as.vector(nv_sqrt(4L)), 2)
  })

  it("keeps a float array's data type", {
    expect_dtype(nv_sqrt(nv_array(4, dtype = "f64")), "f64")
    expect_equal(as.vector(nv_sqrt(nv_array(4, dtype = "f64"))), 2)
  })

  it("honours the default data types", {
    with_default_dtypes(c(float = "f64", int = "i64"), {
      expect_dtype(nv_sqrt(nv_array(4L)), "f64")
    })
  })

  it("does not convert a boolean array", {
    expect_error(nv_sqrt(nv_array(TRUE)))
    expect_error(nv_lgamma(nv_array(TRUE)))
  })
})

describe("nv_atan2", {
  it("converts both operands", {
    out <- nv_atan2(nv_array(1L), nv_array(2L))
    expect_dtype(out, default_float())
    expect_equal(as.vector(out), atan2(1, 2), tolerance = 1e-6)
  })

  it("converts an integer operand to the float the other one brings", {
    # Converting to the default float first and promoting afterwards would
    # round the `i32` through `f32`, which cannot hold 2^24 + 1.
    out <- nv_atan2(nv_array(1, dtype = "f64"), nv_array(16777217L, dtype = "i32"))
    expect_dtype(out, as_dtype("f64"))
    expect_equal(as.vector(out), atan2(1, 16777217), tolerance = 1e-12)
  })
})

describe("nv_floor", {
  it("rounds toward negative infinity", {
    expect_equal(nv_floor(nv_array(c(1.2, 2.7, -1.5))), nv_array(c(1, 2, -2)))
  })

  it("returns an integer array unchanged", {
    expect_equal(nv_floor(nv_array(1:3)), nv_array(1:3))
    expect_equal(nv_floor(nv_array(3L, dtype = "ui8")), nv_array(3L, dtype = "ui8"))
  })

  it("does not accept a boolean array", {
    expect_error(nv_floor(nv_array(TRUE)))
  })
})

describe("nv_ceiling", {
  it("rounds toward positive infinity", {
    expect_equal(nv_ceiling(nv_array(c(1.2, 2.7, -1.5))), nv_array(c(2, 3, -1)))
  })

  it("returns an integer array unchanged", {
    expect_equal(nv_ceiling(nv_array(1:3)), nv_array(1:3))
  })
})

describe("nv_trunc", {
  it("rounds toward zero", {
    expect_equal(
      nv_trunc(nv_array(c(1.2, 2.7, -1.5, -0.3, 0))),
      nv_array(c(1, 2, -1, 0, 0))
    )
  })

  it("returns an integer array unchanged", {
    expect_equal(nv_trunc(nv_array(c(1L, -3L))), nv_array(c(1L, -3L)))
  })
})

describe("nv_round", {
  it("rounds half to even by default and away from zero on request", {
    expect_equal(as.vector(nv_round(nv_array(c(0.5, 1.5)))), round(c(0.5, 1.5)))
    expect_equal(as.vector(nv_round(nv_array(c(0.5, 1.5)), method = "afz")), c(1, 2))
  })

  it("returns an integer array unchanged", {
    expect_equal(nv_round(nv_array(1:3)), nv_array(1:3))
  })
})

describe("nv_sign", {
  it("returns -1, 0 and 1 at the input's data type", {
    expect_equal(nv_sign(nv_array(c(-3, 0, 5))), nv_array(c(-1, 0, 1)))
    expect_equal(nv_sign(nv_array(c(-3L, 0L, 5L))), nv_array(c(-1L, 0L, 1L)))
  })

  it("returns 0 or 1 for an unsigned input, like base R on a non-negative one", {
    x <- nv_array(c(0L, 3L, 7L), dtype = "ui32")
    expect_equal(nv_sign(x), nv_array(c(0L, 1L, 1L), dtype = "ui32"))
    expect_equal(as.vector(nv_sign(x)), sign(c(0L, 3L, 7L)))
    expect_equal(sign(nv_array(0L, dtype = "ui8")), nv_array(0L, dtype = "ui8"))
  })

  it("does not accept a boolean array", {
    expect_error(nv_sign(nv_array(TRUE)))
  })
})

describe("nv_floor_div", {
  it("floors like base R, at both signs and both categories", {
    for (lhs in c(7L, -7L)) {
      for (rhs in c(2L, -2L)) {
        expect_equal(
          as.vector(nv_floor_div(nv_array(lhs), nv_array(rhs))),
          lhs %/% rhs,
          info = sprintf("%d %%/%% %d", lhs, rhs)
        )
        expect_equal(
          as.vector(nv_floor_div(nv_array(as.double(lhs)), nv_array(as.double(rhs)))),
          as.double(lhs) %/% as.double(rhs),
          info = sprintf("%g %%/%% %g", lhs, rhs)
        )
      }
    }
    expect_equal(as.vector(nv_floor_div(nv_array(7.5), nv_array(2.5))), 7.5 %/% 2.5)
  })

  it("works for unsigned integers", {
    expect_equal(
      as.vector(nv_floor_div(nv_array(7L, dtype = "ui8"), nv_array(2L, dtype = "ui8"))),
      7L %/% 2L
    )
  })

  it("agrees with nv_mod(), i.e. (x %/% y) * y + x %% y == x", {
    x <- nv_array(c(7L, -7L, 8L, -8L))
    y <- nv_array(c(3L, 3L, -3L, -3L))
    expect_equal(as.vector(nv_floor_div(x, y) * y + nv_mod(x, y)), as.vector(x))
  })
})

describe("nv_polygamma", {
  it("broadcasts a scalar n", {
    vals <- c(0.5, 1, 2, 5)
    expect_equal(
      nv_polygamma(2, nv_array(vals)),
      nv_array(psigamma(vals, 2)),
      tolerance = 1e-5
    )
    # `n` was static, so an array was refused here while `prim_polygamma()`
    # took one.
    expect_equal(
      nv_polygamma(nv_scalar(2), nv_array(vals)),
      nv_array(psigamma(vals, 2)),
      tolerance = 1e-5
    )
  })

  it("computes an integer array at the default float", {
    out <- nv_polygamma(1, nv_array(1:3))
    expect_dtype(out, default_float())
    expect_equal(as.vector(out), trigamma(1:3), tolerance = 1e-5)
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
  it("is all TRUE for a data type that has no non-finite value, like base R", {
    expect_equal(nv_is_finite(nv_array(1:3)), nv_array(rep(TRUE, 3L)))
    expect_equal(nv_is_finite(nv_array(c(TRUE, FALSE))), nv_array(c(TRUE, TRUE)))
    expect_equal(nv_is_finite(nv_array(1L, dtype = "ui8")), nv_array(TRUE))
    expect_shape(nv_is_finite(nv_array(1:6, shape = c(2L, 3L))), c(2L, 3L))
    # The answer is a constant, so nothing of the input is read.
    hlo <- format(stablehlo(trace_fn(function(a) nv_is_finite(a), list(nv_array(1:3))))[[1L]])
    expect_false(any(grepl("is_finite", hlo, fixed = TRUE)))
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
  it("is all FALSE for a data type that has no NaN, without comparing", {
    expect_equal(nv_is_nan(nv_array(1:3)), nv_array(rep(FALSE, 3L)))
    expect_equal(nv_is_nan(nv_array(c(TRUE, FALSE))), nv_array(c(FALSE, FALSE)))
    expect_equal(nv_is_nan(nv_scalar(1L)), nv_scalar(FALSE))
    expect_shape(nv_is_nan(nv_array(1:6, shape = c(2L, 3L))), c(2L, 3L))
    # The answer is a constant, so nothing of the input is read.
    hlo <- format(stablehlo(trace_fn(function(a) nv_is_nan(a), list(nv_array(1:3))))[[1L]])
    expect_false(any(grepl("compare", hlo, fixed = TRUE)))
  })
})

describe("nv_is_infinite", {
  it("is all FALSE for a data type that has no infinity, without comparing", {
    expect_equal(nv_is_infinite(nv_array(1:3)), nv_array(rep(FALSE, 3L)))
    expect_equal(nv_is_infinite(nv_array(c(TRUE, FALSE))), nv_array(c(FALSE, FALSE)))
    expect_shape(nv_is_infinite(nv_array(1:6, shape = c(2L, 3L))), c(2L, 3L))
    hlo <- format(
      stablehlo(trace_fn(function(a) nv_is_infinite(a), list(nv_array(1:3))))[[1L]]
    )
    expect_false(any(grepl("is_finite", hlo, fixed = TRUE)))
  })
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

describe("reductions with negative dims", {
  it("count from the end", {
    x <- nv_array(array(as.numeric(1:24), c(2, 3, 4)))
    b <- x > 10
    expect_equal(nv_reduce_sum(x, axes = -1L), nv_reduce_sum(x, axes = 3L))
    expect_equal(nv_reduce_prod(x, axes = -2L), nv_reduce_prod(x, axes = 2L))
    expect_equal(nv_reduce_max(x, axes = -3L), nv_reduce_max(x, axes = 1L))
    expect_equal(nv_reduce_min(x, axes = -1L), nv_reduce_min(x, axes = 3L))
    expect_equal(nv_reduce_any(b, axes = -1L), nv_reduce_any(b, axes = 3L))
    expect_equal(nv_reduce_all(b, axes = -1L), nv_reduce_all(b, axes = 3L))
  })
  it("reject out-of-range and duplicated dims", {
    m <- nv_matrix(as.numeric(1:6), nrow = 2)
    expect_error(nv_reduce_sum(m, axes = -3L), "between 1 and 2, or between -2 and -1")
    expect_error(nv_reduce_sum(m, axes = 0L), "between 1 and 2, or between -2 and -1")
    expect_error(nv_reduce_sum(m, axes = c(2L, -1L)), "duplicate axes")
  })
})

describe("nv_reduce_sum / nv_reduce_prod / nv_mean nan_rm", {
  it("propagate NaN by default", {
    x <- nv_array(c(1, NaN, 3, 5))
    expect_true(is.nan(as_array(nv_reduce_sum(x))))
    expect_true(is.nan(as_array(nv_reduce_prod(x))))
    expect_true(is.nan(as_array(nv_mean(x))))
  })
  it("skip NaN when nan_rm = TRUE", {
    v <- c(1, NaN, 3, 5)
    x <- nv_array(v)
    expect_equal(as.numeric(nv_reduce_sum(x, nan_rm = TRUE)), sum(v, na.rm = TRUE))
    expect_equal(as.numeric(nv_reduce_prod(x, nan_rm = TRUE)), prod(v, na.rm = TRUE))
    expect_equal(as.numeric(nv_mean(x, nan_rm = TRUE)), mean(v, na.rm = TRUE))
  })
  it("mean of all-NaN slice returns NaN", {
    x <- nv_array(c(NaN, NaN))
    expect_true(is.nan(as_array(nv_mean(x, nan_rm = TRUE))))
  })
  it("nan_rm forwards from base R generics (sum, prod, mean, range, max, min)", {
    v <- c(1, NaN, 3, 5)
    x <- nv_array(v)
    expect_equal(as.numeric(sum(x, na.rm = TRUE)), sum(v, na.rm = TRUE))
    expect_equal(as.numeric(prod(x, na.rm = TRUE)), prod(v, na.rm = TRUE))
    expect_equal(as.numeric(mean(x, na.rm = TRUE)), mean(v, na.rm = TRUE))
    expect_equal(as.numeric(max(x, na.rm = TRUE)), max(v, na.rm = TRUE))
    expect_equal(as.numeric(min(x, na.rm = TRUE)), min(v, na.rm = TRUE))
    expect_equal(as.numeric(range(x, na.rm = TRUE)), range(v, na.rm = TRUE))
  })
  it("base R generics propagate NaN by default", {
    x <- nv_array(c(1, NaN, 3, 5))
    expect_true(is.nan(as_array(sum(x))))
    expect_true(is.nan(as_array(mean(x))))
    expect_true(is.nan(as_array(max(x))))
  })
})

describe("boolean accumulation in nv_reduce_sum / nv_reduce_prod / nv_cumsum / nv_cumprod", {
  v <- c(TRUE, FALSE, TRUE)
  x <- nv_array(v)

  it("counts a boolean array instead of folding it, like base R", {
    expect_equal(as_array(nv_reduce_sum(x)), sum(v))
    expect_equal(as_array(nv_reduce_prod(x)), prod(v))
    expect_equal(as.numeric(nv_cumsum(x)), as.numeric(cumsum(v)))
    expect_equal(as.numeric(nv_cumprod(x)), as.numeric(cumprod(v)))
  })

  it("accumulates a boolean array at the default integer", {
    expect_dtype(nv_reduce_sum(x), default_int())
    expect_dtype(nv_reduce_prod(x), default_int())
    expect_dtype(nv_cumsum(x), default_int())
    expect_dtype(nv_cumprod(x), default_int())
    with_default_dtypes(c(int = "i64"), {
      expect_dtype(nv_reduce_sum(x), "i64")
      expect_dtype(nv_cumsum(x), "i64")
    })
  })

  it("counts along a single axis", {
    m <- matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
    expect_equal(as.numeric(nv_reduce_sum(nv_array(m), axes = 1L)), as.numeric(colSums(m)))
    expect_equal(as.numeric(nv_reduce_sum(nv_array(m), axes = 2L)), as.numeric(rowSums(m)))
  })

  it("counts through the base R generics", {
    expect_equal(as_array(sum(x)), sum(v))
    expect_equal(as_array(prod(x)), prod(v))
    expect_equal(as.numeric(cumsum(x)), as.numeric(cumsum(v)))
    expect_equal(as.numeric(cumprod(x)), as.numeric(cumprod(v)))
  })

  it("averages a boolean array as a proportion", {
    expect_equal(as.numeric(nv_mean(x)), mean(v), tolerance = 1e-6)
    expect_equal(as.numeric(nv_var(x)), var(v), tolerance = 1e-6)
  })

  it("leaves the data type of a non-boolean input alone", {
    for (dt in c("i32", "i64", "f32", "f64")) {
      y <- nv_array(c(1, 2, 3), dtype = dt)
      expect_dtype(nv_reduce_sum(y), as_dtype(dt))
      expect_dtype(nv_cumsum(y), as_dtype(dt))
    }
  })

  it("leaves the folding reductions boolean", {
    expect_dtype(nv_reduce_any(x), "bool")
    expect_dtype(nv_reduce_all(x), "bool")
    expect_dtype(nv_reduce_max(x), "bool")
    expect_dtype(nv_cummax(x), "bool")
  })

  it("is the nv_* layer's doing -- the primitives keep the StableHLO semantics", {
    expect_dtype(prim_reduce_sum(x, axes = 1L), "bool")
    expect_dtype(prim_cumsum(x, axis = 1L), "bool")
  })
})

describe("nv_var / nv_sd nan_rm", {
  it("propagate NaN by default", {
    x <- nv_array(c(1, NaN, 3, 5))
    expect_true(is.nan(as_array(nv_var(x, axes = 1L))))
    expect_true(is.nan(as_array(nv_sd(x, axes = 1L))))
  })
  it("skip NaN when nan_rm = TRUE (matches base R var/sd)", {
    v <- c(1, NaN, 3, 5)
    x <- nv_array(v)
    expect_equal(as.numeric(nv_var(x, axes = 1L, nan_rm = TRUE)), var(v, na.rm = TRUE), tolerance = 1e-6)
    expect_equal(as.numeric(nv_sd(x, axes = 1L, nan_rm = TRUE)), sd(v, na.rm = TRUE), tolerance = 1e-6)
  })
  it("respects correction argument", {
    v <- c(1, NaN, 3, 5)
    x <- nv_array(v)
    expect_equal(
      as.numeric(nv_var(x, axes = 1L, correction = 0L, nan_rm = TRUE)),
      var(v, na.rm = TRUE) * 2 / 3, # 3 non-NaN values, switch n-1 -> n
      tolerance = 1e-6
    )
  })
  it("all-NaN slice returns NaN, not zero, at default correction = 1", {
    # Regression: previously returned 0 because count - correction = -1
    # and sum_sq / -1 = -0, which coerced to a non-NaN value.
    expect_true(is.nan(as_array(nv_var(nv_array(c(NaN, NaN)), axes = 1L, nan_rm = TRUE))))
    expect_true(is.nan(as_array(nv_sd(nv_array(c(NaN, NaN)), axes = 1L, nan_rm = TRUE))))
    expect_true(is.nan(as_array(nv_var(nv_array(c(NaN, NaN, NaN)), axes = 1L, nan_rm = TRUE))))
  })
  it("count below correction returns NaN; count above is well-defined", {
    # Single non-NaN value with default correction = 1 -> n - 1 = 0 -> NaN
    expect_true(is.nan(as_array(nv_var(nv_array(c(1, NaN)), axes = 1L, nan_rm = TRUE))))
    # Same input with correction = 0 -> population variance of a single
    # value is 0, well-defined.
    expect_equal(as.numeric(nv_var(nv_array(c(1, NaN)), axes = 1L, correction = 0L, nan_rm = TRUE)), 0)
  })
  it("does not let nan_rm change the data type", {
    # The valid-value count is built at the operand's data type, not at an
    # integer one: counting into an integer makes the R double `0` in `nv_max(0,
    # count - correction)` cross categories and materialize the divisor at the
    # default float, so `nan_rm` alone would widen the result.
    x <- nv_array(c(1, 2, NaN, 4), dtype = "f32")
    with_default_dtypes(c(float = "f64", int = "i64"), {
      expect_dtype(nv_var(x, nan_rm = TRUE), dtype(nv_var(x)))
      expect_dtype(nv_var(x, nan_rm = TRUE), "f32")
      expect_dtype(nv_sd(x, nan_rm = TRUE), "f32")
    })
  })

  it("per-slice masking: an all-NaN slice in a matrix yields NaN, others valid", {
    m <- matrix(c(NaN, NaN, 1, 2, 3, 4), nrow = 2) # column 1 all-NaN
    out <- as.numeric(nv_var(nv_array(m), axes = 1L, nan_rm = TRUE))
    expect_true(is.nan(out[1]))
    expect_equal(out[2:3], c(0.5, 0.5))
  })
  it("single-value input with default correction (no NaN) returns NaN", {
    # Matches base R: var(c(1)) is NA. (denom = 0 -> 0/0 = NaN, no change
    # introduced by this commit -- regression check.)
    expect_true(is.nan(as_array(nv_var(nv_array(1.0), axes = 1L))))
    # With correction = 0, population variance of a single value is 0.
    expect_equal(as.numeric(nv_var(nv_array(1.0), axes = 1L, correction = 0L)), 0)
  })
})

describe("nv_range", {
  it("returns the minimum and the maximum as a length-2 array", {
    vals <- c(3, 1, 4, 1, 5, 9, 2, 6)
    expect_equal(nv_range(nv_array(vals)), nv_array(range(vals)))
  })

  it("stacks the two along a new first axis when reducing a single axis", {
    m <- nv_matrix(as.double(1:6), nrow = 2)
    out <- nv_range(m, axes = 1L)
    expect_shape(out, c(2L, 3L))
    expect_equal(as_array(out), rbind(c(1, 3, 5), c(2, 4, 6)))
  })

  it("accepts a negative axis", {
    m <- nv_matrix(as.double(1:6), nrow = 2)
    expect_equal(nv_range(m, axes = -1L), nv_range(m, axes = 2L))
  })

  it("keeps the data type of an integer array", {
    out <- nv_range(nv_array(1:4))
    expect_dtype(out, default_int())
    expect_equal(as.vector(out), c(1L, 4L))
  })

  it("forwards nan_rm", {
    x <- nv_array(c(1, NaN, 3))
    expect_equal(as.vector(nv_range(x, nan_rm = TRUE)), c(1, 3))
    expect_true(all(is.nan(as.vector(nv_range(x)))))
  })
})

describe("nv_reduce_max / nv_reduce_min nan_rm", {
  it("propagates NaN by default (nan_rm = FALSE)", {
    x <- nv_array(c(1, NaN, 3))
    expect_true(is.nan(as_array(nv_reduce_max(x))))
    expect_true(is.nan(as_array(nv_reduce_min(x))))
  })
  it("skips NaN when nan_rm = TRUE", {
    x <- nv_array(c(1, NaN, 3))
    expect_equal(as.numeric(nv_reduce_max(x, nan_rm = TRUE)), 3)
    expect_equal(as.numeric(nv_reduce_min(x, nan_rm = TRUE)), 1)
  })
  it("all-NaN slice returns the identity element when nan_rm = TRUE", {
    x <- nv_array(c(NaN, NaN))
    expect_equal(as.numeric(nv_reduce_max(x, nan_rm = TRUE)), -Inf)
    expect_equal(as.numeric(nv_reduce_min(x, nan_rm = TRUE)), Inf)
  })
  it("propagates per-slice along reduction axes", {
    # column 1 has NaN, columns 2 and 3 do not
    m <- nv_matrix(c(1, NaN, 3, 4, 5, 6), nrow = 2)
    out_default <- as.numeric(nv_reduce_max(m, axes = 1L))
    expect_true(is.nan(out_default[1]))
    expect_equal(out_default[2:3], c(4, 6))
    expect_equal(
      as.numeric(nv_reduce_max(m, axes = 1L, nan_rm = TRUE)),
      c(1, 4, 6)
    )
  })
  it("is a no-op for integer inputs", {
    x <- nv_array(c(1L, 5L, 3L))
    expect_equal(as_array(nv_reduce_max(x)), as_array(nv_reduce_max(x, nan_rm = TRUE)))
    expect_equal(as_array(nv_reduce_min(x)), as_array(nv_reduce_min(x, nan_rm = TRUE)))
  })
})

describe("cumulative ops with a negative dim", {
  it("count from the end", {
    m <- nv_matrix(as.numeric(c(3, 1, 4, 1, 5, 9)), nrow = 2)
    expect_equal(nv_cumsum(m, axis = -1L), nv_cumsum(m, axis = 2L))
    expect_equal(nv_cumprod(m, axis = -2L), nv_cumprod(m, axis = 1L))
    expect_equal(nv_cummax(m, axis = -1L), nv_cummax(m, axis = 2L))
    expect_equal(nv_cummin(m, axis = -1L), nv_cummin(m, axis = 2L))
    expect_error(nv_cumsum(m, axis = 3L), "between 1 and 2, or between -2 and -1")
  })
})

describe("the cumulative ops' axis default", {
  m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
  mr <- as_array(m)

  it("flattens a multi-axis input column-major, like base R", {
    for (fn in list(nv_cumsum, nv_cumprod, nv_cummax, nv_cummin)) {
      expect_equal(as_array(fn(m)), as_array(fn(nv_flatten(m))))
    }
    expect_equal(as.vector(nv_cumsum(m)), cumsum(mr))
    expect_equal(as.vector(nv_cumprod(m)), cumprod(mr))
    expect_equal(as.vector(nv_cummax(m)), cummax(mr))
    expect_equal(as.vector(nv_cummin(m)), cummin(mr))
    expect_shape(nv_cumsum(m), 6L)
  })

  it("accumulates along a chosen axis when one is given", {
    expect_equal(as_array(nv_cumsum(m, axis = 1L)), apply(mr, 2L, cumsum))
    expect_equal(as_array(nv_cumsum(m, axis = 2L)), t(apply(mr, 1L, cumsum)))
  })
})

describe("nv_cumsum / nv_cumprod nan_rm", {
  it("propagates NaN forward by default", {
    x <- nv_array(c(1, NaN, 3))
    out_sum <- as.numeric(nv_cumsum(x))
    expect_equal(out_sum[1], 1)
    expect_true(all(is.nan(out_sum[2:3])))
    out_prod <- as.numeric(nv_cumprod(nv_array(c(2, NaN, 3))))
    expect_equal(out_prod[1], 2)
    expect_true(all(is.nan(out_prod[2:3])))
  })
  it("treats NaN as the identity element when nan_rm = TRUE", {
    expect_equal(as.numeric(nv_cumsum(nv_array(c(1, NaN, 3)), nan_rm = TRUE)), c(1, 1, 4))
    expect_equal(as.numeric(nv_cumprod(nv_array(c(2, NaN, 3)), nan_rm = TRUE)), c(2, 2, 6))
  })
  it("is a no-op for integer inputs", {
    x <- nv_array(c(1L, 2L, 3L))
    expect_equal(as_array(nv_cumsum(x)), as_array(nv_cumsum(x, nan_rm = TRUE)))
    expect_equal(as_array(nv_cumprod(x)), as_array(nv_cumprod(x, nan_rm = TRUE)))
  })
})

describe("nv_cummax / nv_cummin nan_rm", {
  it("propagates NaN forward by default (matches base R)", {
    out_max <- as.numeric(nv_cummax(nv_array(c(1, NaN, 3))))
    expect_equal(out_max[1], 1)
    expect_true(all(is.nan(out_max[2:3])))
    out_min <- as.numeric(nv_cummin(nv_array(c(3, NaN, 1))))
    expect_equal(out_min[1], 3)
    expect_true(all(is.nan(out_min[2:3])))
  })
  it("propagates NaN from the FIRST NaN onwards regardless of later values", {
    # Regression: previously a NaN restarted the cum after itself.
    out <- as.numeric(nv_cummax(nv_array(c(1, 2, NaN, 0, 5))))
    expect_equal(out[1:2], c(1, 2))
    expect_true(all(is.nan(out[3:5])))
  })
  it("skips NaN when nan_rm = TRUE", {
    expect_equal(as.numeric(nv_cummax(nv_array(c(1, NaN, 3)), nan_rm = TRUE)), c(1, 1, 3))
    expect_equal(as.numeric(nv_cummin(nv_array(c(3, NaN, 1)), nan_rm = TRUE)), c(3, 3, 1))
    expect_equal(as.numeric(nv_cummax(nv_array(c(1, 2, NaN, 0, 5)), nan_rm = TRUE)), c(1, 2, 2, 2, 5))
  })
  it("all-NaN slice returns identity at every position when nan_rm = TRUE", {
    x <- nv_array(c(NaN, NaN, NaN))
    expect_equal(as.numeric(nv_cummax(x, nan_rm = TRUE)), c(-Inf, -Inf, -Inf))
    expect_equal(as.numeric(nv_cummin(x, nan_rm = TRUE)), c(Inf, Inf, Inf))
  })
  it("is a no-op for integer inputs", {
    x <- nv_array(c(3L, 1L, 4L))
    expect_equal(as_array(nv_cummax(x)), as_array(nv_cummax(x, nan_rm = TRUE)))
    expect_equal(as_array(nv_cummin(x)), as_array(nv_cummin(x, nan_rm = TRUE)))
  })
  it("indices returns NaN-propagated values and indices", {
    out <- nv_cummax(nv_array(c(1, NaN, 3)), indices = TRUE)
    vals <- as.numeric(out$values)
    expect_equal(vals[1], 1)
    expect_true(all(is.nan(vals[2:3])))
    # Once NaN propagates, the index tiebreak in the reducer picks the
    # larger index, which for cumulative scans is the current position.
    expect_equal(as.integer(out$indices), c(1L, 2L, 3L))
  })
})

describe("nv_argmax / nv_argmin nan_rm", {
  it("propagates NaN by default (nan_rm = FALSE): returns the NaN's index", {
    x <- nv_array(c(1, NaN, 3))
    expect_equal(as.integer(nv_argmax(x)), 2L)
    expect_equal(as.integer(nv_argmin(x)), 2L)
  })
  it("skips NaN when nan_rm = TRUE", {
    x <- nv_array(c(1, NaN, 3))
    expect_equal(as.integer(nv_argmax(x, nan_rm = TRUE)), 3L)
    expect_equal(as.integer(nv_argmin(x, nan_rm = TRUE)), 1L)
  })
  it("returns first NaN when several NaNs exist", {
    x <- nv_array(c(1, NaN, 3, NaN, 5))
    expect_equal(as.integer(nv_argmax(x)), 2L)
    expect_equal(as.integer(nv_argmin(x)), 2L)
  })
  it("propagates per-slice along the reduced axis", {
    # row 1 has NaN at col 2, row 2 has no NaN
    m <- nv_matrix(c(1, NaN, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(as.integer(nv_argmax(m, axes = 2L)), c(2L, 2L))
    expect_equal(
      as.integer(nv_argmax(m, axes = 2L, nan_rm = TRUE)),
      c(3L, 2L)
    )
  })
})

describe("nv_var", {
  it("computes variance with Bessel's correction", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_equal(
      nv_var(nv_array(vals), axes = 1L),
      nv_scalar(var(vals)),
      tolerance = 1e-5
    )
  })
  it("computes population variance with correction = 0", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expected <- mean((vals - mean(vals))^2)
    expect_equal(
      nv_var(nv_array(vals), axes = 1L, correction = 0L),
      nv_scalar(expected),
      tolerance = 1e-5
    )
  })
  it("works along specific axes of a matrix", {
    vals <- c(1, 2, 3, 4, 5, 6)
    m <- matrix(vals, nrow = 2)
    expect_equal(
      nv_var(nv_array(vals, shape = c(2, 3), dtype = "f32"), axes = 2L),
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
      nv_var(nv_array(vals), axes = NULL),
      nv_scalar(var(vals)),
      tolerance = 1e-5
    )
  })
  it("rejects out-of-range and duplicate axes", {
    expect_error(nv_var(nv_array(c(1, 2, 3, 4)), axes = 5L))
    expect_error(nv_var(nv_array(c(1, 2, 3, 4)), axes = c(1L, 1L)))
  })
  it("accepts negative dims", {
    m <- nv_matrix(c(2, 4, 4, 4, 5, 5), nrow = 2)
    expect_equal(nv_var(m, axes = -1L), nv_var(m, axes = 2L))
    expect_error(nv_var(m, axes = c(2L, -1L)), "duplicate axes")
  })
})

describe("nv_sd", {
  it("computes standard deviation", {
    vals <- c(2, 4, 4, 4, 5, 5, 7, 9)
    expect_equal(
      nv_sd(nv_array(vals), axes = 1L),
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
      nv_sd(nv_array(vals), axes = NULL),
      nv_scalar(sd(vals)),
      tolerance = 1e-5
    )
  })
  it("accepts negative dims", {
    m <- nv_matrix(c(2, 4, 4, 4, 5, 5), nrow = 2)
    expect_equal(nv_sd(m, axes = -1L), nv_sd(m, axes = 2L))
  })
})

describe("nv_squeeze", {
  it("removes all size-1 axes by default", {
    expect_equal(
      {
        x <- nv_array(1:6, shape = c(1, 6, 1))
        nv_squeeze(x)
      },
      nv_array(1:6, shape = 6L)
    )
  })
  it("removes specific axes", {
    expect_equal(
      {
        x <- nv_array(1:6, shape = c(1, 6, 1))
        nv_squeeze(x, axes = 1L)
      },
      nv_array(1:6, shape = c(6, 1))
    )
  })
  it("errors when squeezing non-1 axis", {
    expect_error(
      nv_squeeze(nv_array(1:6, shape = c(2, 3)), axes = 1L),
      "Cannot squeeze"
    )
  })
  it("rejects duplicate axes", {
    expect_error(
      nv_squeeze(nv_array(1:6, shape = c(1, 6, 1)), axes = c(1L, 1L))
    )
  })
  it("accepts negative dims", {
    x <- nv_array(1:6, shape = c(1, 6, 1))
    expect_equal(nv_squeeze(x, axes = -1L), nv_squeeze(x, axes = 3L))
    expect_error(nv_squeeze(x, axes = -4L), "between 1 and 3, or between -3 and -1")
  })
})

describe("nv_unsqueeze", {
  it("adds axis at the beginning", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3))
        nv_unsqueeze(x, axis = 1L)
      },
      nv_array(c(1, 2, 3), shape = c(1, 3))
    )
  })
  it("adds axis at the end", {
    expect_equal(
      {
        x <- nv_array(c(1, 2, 3))
        nv_unsqueeze(x, axis = 2L)
      },
      nv_array(c(1, 2, 3), shape = c(3, 1))
    )
  })
  it("adds axis in the middle", {
    x <- nv_array(1:6, shape = c(2, 3))
    result <- nv_unsqueeze(x, axis = 2L)
    expect_shape(result, c(2L, 1L, 3L))
    roundtrip <- nv_squeeze(nv_unsqueeze(x, axis = 2L), axes = 2L)
    expect_equal(roundtrip, x)
  })
  it("counts negative dims from the end of the result", {
    x <- nv_array(c(1, 2, 3))
    expect_shape(nv_unsqueeze(x, axis = -1L), c(3L, 1L))
    expect_shape(nv_unsqueeze(x, axis = -2L), c(1L, 3L))
  })
  it("allows inserting one past the end but no further", {
    m <- nv_array(1:6, shape = c(2, 3))
    expect_shape(nv_unsqueeze(m, axis = 3L), c(2L, 3L, 1L))
    expect_error(nv_unsqueeze(m, axis = 4L), "between 1 and 3")
    expect_error(nv_unsqueeze(m, axis = c(1L, 2L)), "must have length 1")
  })
})

describe("nv_linspace", {
  it("creates evenly spaced values", {
    expect_equal(
      nv_linspace(0, 1, steps = 5L),
      nv_array(c(0, 0.25, 0.5, 0.75, 1)),
      tolerance = 1e-6
    )
  })
  it("handles single step", {
    expect_equal(
      nv_linspace(3, 7, steps = 1L),
      nv_array(3, shape = 1L),
      tolerance = 1e-6
    )
  })
  it("works with integer-like endpoints", {
    expect_equal(
      nv_linspace(0, 10, steps = 6L),
      nv_array(c(0, 2, 4, 6, 8, 10)),
      tolerance = 1e-6
    )
  })
  it("counts down when end is below start", {
    expect_equal(
      nv_linspace(1, 0, steps = 5L),
      nv_array(c(1, 0.75, 0.5, 0.25, 0)),
      tolerance = 1e-6
    )
  })
  it("defaults to the default float dtype", {
    expect_dtype(nv_linspace(0, 1, steps = 3L), default_float())
    expect_dtype(nv_linspace(0, 1, steps = 1L), default_float())
    with_default_dtypes(c(float = "f64"), {
      expect_dtype(nv_linspace(0, 1, steps = 3L), "f64")
    })
  })
  it("honours a float dtype", {
    expect_dtype(nv_linspace(0, 1, steps = 3L, dtype = "f64"), "f64")
    expect_dtype(nv_linspace(0, 1, steps = 1L, dtype = "f64"), "f64")
  })
  it("rejects an integer dtype", {
    expect_error(nv_linspace(0, 1, steps = 5L, dtype = "i32"), "must be a float data type")
    expect_error(nv_linspace(0, 10, steps = 6L, dtype = "i32"), "must be a float data type")
    expect_error(nv_linspace(0, 1, steps = 1L, dtype = "i32"), "must be a float data type")
  })
  it("requires steps to be a positive whole number", {
    expect_error(nv_linspace(0, 1, steps = 0L), "steps")
    expect_error(nv_linspace(0, 1, steps = 2.5), "steps")
  })
})

describe("nv_seq", {
  it("creates consecutive integer values", {
    expect_equal(nv_seq(3, 7), nv_array(3:7))
  })
  it("defaults to the default integer dtype", {
    expect_dtype(nv_seq(3, 7), default_int())
    with_default_dtypes(c(int = "i64"), expect_dtype(nv_seq(3, 7), "i64"))
  })
  it("no longer takes steps", {
    expect_error(nv_seq(0, 1, steps = 5L), "unused argument")
  })
  it("counts down when start is greater than end", {
    expect_equal(nv_seq(7, 3), nv_array(7:3))
  })
  it("steps by `by`", {
    expect_equal(nv_seq(0, 10, by = 2), nv_array(seq(0L, 10L, by = 2L)))
    expect_equal(nv_seq(10, 0, by = -3), nv_array(seq(10L, 0L, by = -3L)))
  })
  it("stops before end when end is not reachable", {
    expect_equal(nv_seq(0, 9, by = 2), nv_array(seq(0L, 9L, by = 2L)))
  })
  it("returns a single value when start equals end", {
    expect_equal(nv_seq(3, 3), nv_array(3L))
    expect_equal(nv_seq(3, 3, by = -2), nv_array(3L))
  })
  it("keeps the default integer dtype when stepping", {
    expect_dtype(nv_seq(0, 10, by = 2), default_int())
    with_default_dtypes(c(int = "i64"), expect_dtype(nv_seq(0, 10, by = 2), "i64"))
    expect_dtype(nv_seq(0, 10, by = 2, dtype = "i16"), "i16")
  })
  it("errors for a zero, fractional or wrongly signed `by`", {
    expect_error(nv_seq(0, 10, by = 0), "must not be 0")
    expect_error(nv_seq(0, 10, by = 2.5), "by")
    expect_error(nv_seq(0, 10, by = -2), "Wrong sign")
    expect_error(nv_seq(10, 0, by = 2), "Wrong sign")
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
  it("rejects non-1-D inputs", {
    expect_error(nv_diag(nv_matrix(1:6, nrow = 2)), "1-D array")
    expect_error(nv_diag(nv_scalar(5)), "1-D array")
  })
})

describe("nv_lower_tri / nv_upper_tri", {
  # The default `diagonal` excludes the main diagonal, like base R's
  # `diag = FALSE`; `diagonal = 0L` includes it, like `diag = TRUE`. Those are
  # the only two offsets base R's logical `diag` can express.
  expect_matches_base <- function(nv_fn, base_fn, default_diagonal) {
    for (shape in list(c(3, 3), c(2, 5), c(5, 2))) {
      x <- matrix(0, shape[1L], shape[2L])
      expect_equal(as_array(nv_fn(shape)), base_fn(x))
      expect_equal(as_array(nv_fn(shape, diagonal = default_diagonal)), base_fn(x))
      expect_equal(as_array(nv_fn(shape, diagonal = 0L)), base_fn(x, diag = TRUE))
    }
  }

  it("nv_lower_tri matches base R lower.tri()", {
    expect_matches_base(nv_lower_tri, lower.tri, -1L)
  })
  it("nv_upper_tri matches base R upper.tri()", {
    expect_matches_base(nv_upper_tri, upper.tri, 1L)
  })
  it("supports offsets base R's logical diag cannot express", {
    expect_equal(
      as_array(nv_lower_tri(c(4, 4), diagonal = 2L)),
      outer(1:4, 1:4, function(i, j) i >= j - 2L)
    )
    expect_equal(
      as_array(nv_upper_tri(c(4, 4), diagonal = 2L)),
      outer(1:4, 1:4, function(i, j) i <= j - 2L)
    )
  })
  it("returns bool", {
    expect_dtype(nv_lower_tri(c(3, 3)), "bool")
    expect_dtype(nv_upper_tri(c(3, 3)), "bool")
  })
  it("rejects a shape that is not 2-D", {
    expect_error(nv_lower_tri(3), "must have length 2")
    expect_error(nv_upper_tri(3), "must have length 2")
  })
  it("rejects a non-integer diagonal", {
    expect_error(nv_lower_tri(c(3, 3), diagonal = "a"))
    expect_error(nv_upper_tri(c(3, 3), diagonal = "a"))
  })
  it("the _like variants inherit shape from like and stay bool", {
    x <- nv_fill(0, c(4, 2), dtype = "f64")
    expect_equal(as_array(nv_lower_tri_like(x)), lower.tri(matrix(0, 4, 2)))
    expect_equal(as_array(nv_upper_tri_like(x)), upper.tri(matrix(0, 4, 2)))
    expect_dtype(nv_lower_tri_like(x), "bool")
    expect_shape(nv_lower_tri_like(x, shape = c(2, 2)), c(2L, 2L))
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
  it("rejects a non-integer diagonal", {
    expect_error(nv_tril(nv_fill(1, c(3, 3)), diagonal = "a"))
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
  it("rejects a non-integer diagonal", {
    expect_error(nv_triu(nv_fill(1, c(3, 3)), diagonal = "a"))
  })
})

describe("nv_tril with quickr backend", {
  it("works when the input is quickr", {
    skip_if_no_quickr()
    local_backend("quickr")
    x <- nv_matrix(1, nrow = 3, ncol = 3)
    result <- nv_tril(x)
    expected <- matrix(c(1, 1, 1, 0, 1, 1, 0, 0, 1), nrow = 3, ncol = 3)
    expect_equal(as_array(result), expected, tolerance = 1e-6)
  })
})

describe("nv_triu with quickr backend", {
  it("works when the input is quickr", {
    skip_if_no_quickr()
    local_backend("quickr")
    x <- nv_matrix(1, nrow = 3, ncol = 3)
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
  it("transposes only the matrix axes of a batched array", {
    a <- array(as.numeric(1:12), c(2, 3, 2))
    out <- as_array(nv_crossprod(nv_array(a, dtype = "f64")))
    expect_equal(dim(out), c(2L, 2L, 2L))
    expect_equal(out[1, , ], crossprod(a[1, , ]))
    expect_equal(out[2, , ], crossprod(a[2, , ]))
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
  it("transposes only the matrix axes of a batched array", {
    a <- array(as.numeric(1:12), c(2, 3, 2))
    out <- as_array(nv_tcrossprod(nv_array(a, dtype = "f64")))
    expect_equal(dim(out), c(2L, 3L, 3L))
    expect_equal(out[1, , ], tcrossprod(a[1, , ]))
    expect_equal(out[2, , ], tcrossprod(a[2, , ]))
  })
})

describe("nv_fill_like", {
  it("inherits shape, dtype, device from like", {
    like <- nv_matrix(1:6, nrow = 2, dtype = "i16")
    out <- nv_fill_like(like, 0L)
    expect_shape(out, shape(like))
    expect_dtype(out, dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
    expect_equal(as.integer(out), rep(0L, 6L))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_matrix(1:6, nrow = 2, dtype = "i16")
    out <- nv_fill_like(like, 1, shape = 5L, dtype = "f32")
    expect_shape(out, 5L)
    expect_dtype(out, "f32")
  })
})

describe("nv_iota_like", {
  it("inherits shape, dtype, device from like", {
    like <- nv_fill(0L, shape = c(2, 3), dtype = "i16")
    out <- nv_iota_like(like, axis = 1L)
    expect_shape(out, shape(like))
    expect_dtype(out, dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_fill(0L, shape = c(2, 3), dtype = "i16")
    out <- nv_iota_like(like, axis = 1L, shape = 4L, dtype = "i32")
    expect_shape(out, 4L)
    expect_dtype(out, "i32")
  })

  it("accepts a negative dim", {
    like <- nv_fill(0L, shape = c(2, 3), dtype = "i16")
    expect_equal(nv_iota_like(like, axis = -1L), nv_iota_like(like, axis = 2L))
  })
})

describe("nv_seq_like", {
  it("inherits dtype, device from like (length determined by start/end)", {
    like <- nv_array(c(0L, 0L, 0L), dtype = "i16")
    out <- nv_seq_like(like, 1L, 5L)
    expect_dtype(out, dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
    expect_shape(out, 5L)
    expect_equal(as.integer(out), c(1L, 2L, 3L, 4L, 5L))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_array(c(0L, 0L, 0L), dtype = "i16")
    out <- nv_seq_like(like, 1, 5, dtype = "f32")
    expect_dtype(out, "f32")
  })

  it("passes `by` through", {
    like <- nv_array(c(0L, 0L, 0L), dtype = "i16")
    out <- nv_seq_like(like, 0, 10, by = 5)
    expect_dtype(out, "i16")
    expect_equal(as.integer(out), c(0L, 5L, 10L))
  })
})

describe("nv_linspace_like", {
  it("inherits dtype, device from like (length determined by steps)", {
    like <- nv_array(c(0, 0, 0), dtype = "f64")
    out <- nv_linspace_like(like, 0, 1, steps = 5L)
    expect_dtype(out, dtype(like))
    expect_equal(as.character(device(out)), as.character(device(like)))
    expect_shape(out, 5L)
    expect_equal(as.numeric(out), c(0, 0.25, 0.5, 0.75, 1))
  })

  it("allows overriding the inherited attributes", {
    like <- nv_array(c(0, 0, 0), dtype = "f64")
    expect_dtype(nv_linspace_like(like, 0, 1, steps = 3L, dtype = "f32"), "f32")
  })

  it("rejects an integer like", {
    like <- nv_array(c(0L, 0L, 0L), dtype = "i16")
    expect_error(nv_linspace_like(like, 0, 1, steps = 5L), "must be a float data type")
  })
})

describe("nv_select", {
  it("selects a row of a matrix and drops the axis", {
    m <- nv_matrix(1:6, nrow = 2)
    expect_equal(nv_select(m, axis = 1L, index = 1L), nv_array(c(1L, 3L, 5L)))
    expect_equal(nv_select(m, axis = 1L, index = 2L), nv_array(c(2L, 4L, 6L)))
  })

  it("selects a column of a matrix and drops the axis", {
    m <- nv_matrix(1:6, nrow = 2)
    expect_equal(nv_select(m, axis = 2L, index = 2L), nv_array(c(3L, 4L)))
  })

  it("array(i) keeps the axis with size 1", {
    x <- nv_array(1:6, shape = c(2L, 3L))
    out <- nv_select(x, axis = 2L, index = array(1L))
    expect_shape(out, c(2L, 1L))
  })

  it("works on a 3D array", {
    arr <- nv_array(1:24, shape = c(2, 3, 4))
    out <- nv_select(arr, axis = 3L, index = 2L)
    expect_shape(out, c(2L, 3L))
    expect_equal(as_array(out), array(7:12, dim = c(2, 3)))
  })

  it("errors when axis is out of bounds", {
    expect_error(nv_select(nv_array(c(1, 2, 3)), axis = 2L, index = 1L))
  })

  it("errors when index is out of bounds", {
    expect_error(nv_select(nv_array(c(1, 2, 3)), axis = 1L, index = 5L))
  })

  it("errors on a 0-dimensional input", {
    expect_error(nv_select(nv_scalar(1), axis = 1L, index = 1L), "at least one axis")
  })

  it("accepts a negative dim", {
    m <- nv_matrix(1:6, nrow = 2)
    expect_equal(nv_select(m, axis = -1L, index = 2L), nv_select(m, axis = 2L, index = 2L))
    expect_error(nv_select(m, axis = -3L, index = 1L), "between 1 and 2, or between -2 and -1")
  })
})

describe("nv_sort", {
  it("defaults axis to the last axis", {
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

  it("flattens a matrix by default, like base R", {
    mr <- matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    m <- nv_array(mr)
    expect_equal(as.vector(nv_sort(m)), sort(mr))
    expect_shape(nv_sort(m), 6L)
    # `sort()` on an anvl array agrees with base R exactly
    expect_equal(as.vector(sort(m)), sort(mr))
  })

  it("sorts each slice when an axis is given", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expected <- nv_matrix(c(1, 3, 5, 0, 2, 4), nrow = 2, byrow = TRUE)
    expect_equal(nv_sort(m, axis = 2L), expected)
  })

  it("errors on a 0-dimensional input", {
    expect_error(nv_sort(nv_scalar(1)), "at least one axis")
  })

  it("dispatches via the sort() generic", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    expect_equal(as.vector(sort(x)), c(1, 1, 3, 4, 5))
    expect_equal(as.vector(sort(x, decreasing = TRUE)), c(5, 4, 3, 1, 1))
    expect_equal(as.vector(jit(function(x) sort(x))(x)), c(1, 1, 3, 4, 5))
  })

  it("accepts a negative dim", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(nv_sort(m, axis = -2L), nv_sort(m, axis = 1L))
    expect_error(nv_sort(m, axis = 0L), "between 1 and 2, or between -2 and -1")
  })
})

describe("nv_argsort", {
  it("returns indices that sort the array", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    perm <- as.integer(nv_argsort(x))
    expect_equal(as.vector(x)[perm], c(1, 1, 3, 4, 5))
  })

  it("supports decreasing", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    perm <- as.integer(nv_argsort(x, decreasing = TRUE))
    expect_equal(as.vector(x)[perm], c(5, 4, 3, 1, 1))
  })

  it("returns i32 dtype", {
    expect_dtype(nv_argsort(nv_array(c(1, 2))), default_int())
  })

  it("accepts a negative dim", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(nv_argsort(m, axis = -2L), nv_argsort(m, axis = 1L))
  })

  it("flattens a matrix by default, matching nv_sort", {
    mr <- matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    m <- nv_array(mr)
    perm <- as.integer(nv_argsort(m))
    expect_shape(nv_argsort(m), 6L)
    # the indices refer to the column-major flattening, which is what nv_sort
    # sorts, so indexing it by them reproduces nv_sort()'s output
    expect_equal(as.vector(mr)[perm], as.vector(nv_sort(m)))
    # which is base R's order(): `mr` has no ties, so the permutation is unique
    expect_equal(perm, order(mr))
    expect_equal(as.integer(nv_argsort(m, decreasing = TRUE)), order(mr, decreasing = TRUE))
  })

  it("permutes each slice when an axis is given", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_shape(nv_argsort(m, axis = 2L), c(2L, 3L))
    expect_equal(nv_argsort(m, axis = 2L), nv_argsort(m, axis = -1L))
  })
})

describe("nv_top_k", {
  it("returns the k largest values of a 1-D array", {
    expect_equal(
      nv_top_k(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)), k = 3L),
      nv_array(c(9, 6, 5))
    )
  })

  it("operates per-row on a matrix when given the last axis", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    out <- nv_top_k(m, k = 2L, axes = 2L)
    expect_shape(out, c(2L, 2L))
    expect_equal(as_array(out), matrix(c(5, 3, 4, 2), nrow = 2, byrow = TRUE))
  })

  it("puts the k axis where the first reduced axis was", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    out <- nv_top_k(m, k = 1L, axes = 1L)
    expect_shape(out, c(1L, 3L))
    expect_equal(as_array(out), matrix(c(3, 4, 5), nrow = 1))
  })

  it("ranks every axis together by default, like flattening first", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(nv_top_k(m, k = 3L), nv_array(c(5, 4, 3)))
    expect_equal(nv_top_k(m, k = 3L), nv_top_k(nv_flatten(m), k = 3L))
    expect_equal(nv_top_k(m, k = 3L, axes = c(1L, 2L)), nv_top_k(m, k = 3L))
  })

  it("indexes the column-major flattening of the reduced axes", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    # Along one axis the index is a position in that axis.
    expect_equal(
      as_array(nv_top_k(m, k = 2L, axes = 2L, indices = TRUE)$indices),
      matrix(c(3L, 1L, 2L, 1L), nrow = 2, byrow = TRUE)
    )
    # Over both, it is a position in `nv_flatten(m)`.
    # nv_flatten(m) is 3, 2, 1, 4, 5, 0, so 5, 4 and 3 sit at 5, 4 and 1
    out <- nv_top_k(m, k = 3L, indices = TRUE)
    expect_equal(as.integer(as_array(out$indices)), c(5L, 4L, 1L))
    expect_equal(
      as.vector(as_array(out$values)),
      as.vector(as_array(nv_flatten(m)))[c(5L, 4L, 1L)]
    )
  })

  it("errors when k exceeds what the reduced axes hold", {
    expect_error(nv_top_k(nv_array(c(1, 2, 3)), k = 5L))
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_error(nv_top_k(m, k = 4L, axes = 2L), "The ranked axis holds 3 of them")
    # Over both axes the same `k` is fine, since they hold six elements.
    expect_error(nv_top_k(m, k = 4L), NA)
    expect_error(nv_top_k(m, k = 7L), "The ranked axes hold 6 of them")
  })

  it("accepts a negative axis", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(nv_top_k(m, k = 2L, axes = -1L), nv_top_k(m, k = 2L, axes = 2L))
    expect_equal(nv_top_k(m, k = 2L, axes = -2L), nv_top_k(m, k = 2L, axes = 1L))
  })
})

describe("nv_median / nv_quantile NaN handling", {
  it("propagates NaN regardless of NaN position", {
    # All three should be NaN -- previously some returned non-NaN because
    # XLA's sort placed NaN unpredictably and the lo/hi gather missed it.
    expect_true(is.nan(as_array(nv_median(nv_array(c(1, NaN, 3, 5))))))
    expect_true(is.nan(as_array(nv_median(nv_array(c(NaN, 1, 3, 5))))))
    expect_true(is.nan(as_array(nv_median(nv_array(c(1, 3, 5, NaN))))))
  })
  it("propagates NaN in quantile for all probs and interpolations", {
    x <- nv_array(c(1, NaN, 3, 5))
    for (q in c(0, 0.25, 0.5, 0.75, 1)) {
      expect_true(is.nan(as_array(nv_quantile(x, q))), info = paste("q =", q))
    }
    for (method in c("linear", "lower", "higher", "nearest", "midpoint")) {
      expect_true(is.nan(as_array(nv_quantile(x, 0.5, interpolation = method))), info = paste("method =", method))
    }
  })
  it("propagates NaN for array probs", {
    out <- as.numeric(nv_quantile(nv_array(c(1, NaN, 3, 5)), array(c(0.25, 0.5, 0.75))))
    expect_true(all(is.nan(out)))
  })
  it("does not let nan_rm change the data type", {
    # Both branches of the valid-value count are built at `dtype(x)`, so the `-
    # 1` in `(n_valid - 1) * probs` yields to it. An integer count there would
    # cross categories and materialize `h` -- and with it `lo_f`, `frac` and the
    # result -- at the default float.
    x <- nv_array(c(1, 2, NaN, 4), dtype = "f32")
    with_default_dtypes(c(float = "f64", int = "i64"), {
      expect_dtype(nv_quantile(x, 0.5, nan_rm = TRUE), dtype(nv_quantile(x, 0.5)))
      expect_dtype(nv_quantile(x, 0.5, nan_rm = TRUE), "f32")
      expect_dtype(nv_median(x, nan_rm = TRUE), "f32")
      # Array `probs` takes the same path.
      expect_dtype(nv_quantile(x, array(c(0.25, 0.75)), nan_rm = TRUE), "f32")
    })
  })

  it("nan_rm = TRUE matches base R quantile", {
    v <- c(1, NaN, 3, 5)
    x <- nv_array(v)
    for (q in c(0, 0.25, 0.5, 0.75, 1)) {
      expect_equal(
        as.numeric(nv_quantile(x, q, nan_rm = TRUE)),
        unname(quantile(v, q, na.rm = TRUE)),
        info = paste("q =", q),
        tolerance = 1e-6
      )
    }
  })
  it("nan_rm = TRUE matches base R for all interpolations", {
    # Use 4 valid values so q = 0.5 falls between two indices (h = 1.5)
    v <- c(1, 2, 3, NaN, 5)
    x <- nv_array(v)
    expect_equal(
      as.numeric(nv_quantile(x, 0.5, interpolation = "linear", nan_rm = TRUE)),
      median(v, na.rm = TRUE)
    )
    expect_equal(as.numeric(nv_quantile(x, 0.5, interpolation = "lower", nan_rm = TRUE)), 2)
    expect_equal(as.numeric(nv_quantile(x, 0.5, interpolation = "higher", nan_rm = TRUE)), 3)
    expect_equal(as.numeric(nv_quantile(x, 0.5, interpolation = "midpoint", nan_rm = TRUE)), 2.5)
  })
  it("nan_rm = TRUE with array probs returns one quantile per prob", {
    v <- c(1, NaN, 3, 5)
    x <- nv_array(v)
    expect_equal(
      as.numeric(nv_quantile(x, array(c(0.25, 0.5, 0.75)), nan_rm = TRUE)),
      unname(quantile(v, c(0.25, 0.5, 0.75), na.rm = TRUE)),
      tolerance = 1e-6
    )
  })
  it("all-NaN slice returns NaN with nan_rm = TRUE", {
    expect_true(is.nan(as_array(nv_median(nv_array(c(NaN, NaN)), nan_rm = TRUE))))
    expect_true(is.nan(as_array(nv_quantile(nv_array(c(NaN, NaN)), 0.5, nan_rm = TRUE))))
  })
  it("nv_median forwards nan_rm", {
    expect_equal(as.numeric(nv_median(nv_array(c(1, NaN, 3, 5)), nan_rm = TRUE)), 3)
  })
  it("median() generic forwards na.rm", {
    expect_equal(as.numeric(median(nv_array(c(1, NaN, 3, 5)), na.rm = TRUE)), 3)
    expect_true(is.nan(as_array(median(nv_array(c(1, NaN, 3, 5))))))
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

  it("reduces every axis of a matrix by default, like base R", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(as.vector(nv_median(m)), median(c(3, 1, 5, 2, 4, 0)))
    expect_shape(nv_median(m), integer())
  })

  it("reduces the named axes only", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(as.vector(nv_median(m, axes = 2L)), c(3, 2))
    expect_equal(as.vector(nv_median(m, axes = 1L)), c(2.5, 2.5, 2.5))
    expect_equal(nv_median(m, axes = c(1L, 2L)), nv_median(m))
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

  it("accepts negative axes", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(nv_median(m, axes = -1L), nv_median(m, axes = 2L))
  })

  it("keeps the reduced axes at size 1 when drop = FALSE", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_shape(nv_median(m, axes = 2L, drop = FALSE), c(2L, 1L))
    expect_shape(nv_median(m, drop = FALSE), c(1L, 1L))
    expect_equal(
      as.vector(nv_median(m, axes = 2L, drop = FALSE)),
      as.vector(nv_median(m, axes = 2L))
    )
  })

  it("computes a non-float array at the default float, like base R", {
    expect_equal(as.vector(nv_median(nv_array(1:4))), median(1:4))
    expect_equal(as.vector(nv_median(nv_array(c(TRUE, FALSE)))), median(c(TRUE, FALSE)))
    expect_dtype(nv_median(nv_array(1:4)), default_float())
  })

  it("honours the default data types", {
    with_default_dtypes(c(float = "f64", int = "i64"), {
      out <- nv_median(nv_array(1:4))
      expect_dtype(out, "f64")
      expect_equal(as.vector(out), median(1:4))
    })
  })

  it("keeps a float array's data type", {
    x <- nv_array(c(1, 2, 3, 4), dtype = "f64")
    expect_dtype(nv_median(x), "f64")
    expect_equal(as.vector(nv_median(x)), median(c(1, 2, 3, 4)))
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

  it("vector probs prepends a leading axis of length(probs)", {
    xr <- c(3, 1, 4, 1, 5, 9, 2, 6)
    x <- nv_array(xr)
    out <- nv_quantile(x, array(c(0.25, 0.5, 0.75)))
    expect_shape(out, 3L)
    expect_equal(as.vector(out), unname(quantile(xr, c(0.25, 0.5, 0.75))))
  })

  it("vector probs work for >1-D inputs (frac broadcast)", {
    mr <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 7, 0, 5, 4), nrow = 3)
    m <- nv_array(mr)
    out <- nv_quantile(m, array(c(0.25, 0.75)), axes = 2L)
    expect_shape(out, c(2L, 3L))
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

  it("operates along a chosen axis of a matrix", {
    m_raw <- matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    m <- nv_array(m_raw)
    out <- nv_quantile(m, 0.5, axes = 2L)
    expect_equal(as.vector(out), c(3, 2))
  })

  it("reduces every axis by default, like base R", {
    mr <- matrix(c(3, 1, 4, 1, 5, 9, 2, 6, 7, 0, 5, 4), nrow = 3)
    m <- nv_array(mr)
    for (q in c(0, 0.25, 0.5, 0.75, 1)) {
      expect_equal(as_array(nv_quantile(m, q)), unname(quantile(mr, q)), info = paste("q =", q))
    }
    expect_shape(nv_quantile(m, 0.5), integer())
  })

  it("ranks the elements of several axes together", {
    a <- nv_array(as.numeric(1:24), shape = c(2L, 3L, 4L))
    ar <- as_array(a)
    out <- nv_quantile(a, 0.5, axes = c(1L, 3L))
    expect_shape(out, 3L)
    expect_equal(as.vector(out), apply(ar, 2L, median))
    # reducing every axis is the same as flattening first
    expect_equal(nv_quantile(a, 0.5, axes = c(1L, 2L, 3L)), nv_quantile(nv_flatten(a), 0.5))
  })

  it("keeps the reduced axes at size 1 when drop = FALSE", {
    a <- nv_array(as.numeric(1:24), shape = c(2L, 3L, 4L))
    expect_shape(nv_quantile(a, 0.5, axes = c(1L, 3L), drop = FALSE), c(1L, 3L, 1L))
    expect_equal(
      as.vector(nv_quantile(a, 0.5, axes = c(1L, 3L), drop = FALSE)),
      as.vector(nv_quantile(a, 0.5, axes = c(1L, 3L)))
    )
  })

  it("prepends the probs axis in front of the kept axes for several axes", {
    a <- nv_array(as.numeric(1:24), shape = c(2L, 3L, 4L))
    ar <- as_array(a)
    out <- nv_quantile(a, array(c(0.25, 0.75)), axes = c(1L, 3L))
    expect_shape(out, c(2L, 3L))
    expect_equal(as_array(out), apply(ar, 2L, quantile, probs = c(0.25, 0.75)), ignore_attr = TRUE)
  })

  it("reduces nothing for axes = integer()", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(as_array(nv_quantile(m, 0.5, axes = integer())), as_array(m))
  })

  it("rejects probs outside [0, 1]", {
    expect_error(nv_quantile(nv_array(c(1, 2)), -0.1))
    expect_error(nv_quantile(nv_array(c(1, 2)), 1.5))
  })

  it("returns a scalar input unchanged, like the other reductions", {
    expect_equal(as_array(nv_quantile(nv_scalar(7), 0.5)), as_array(nv_scalar(7)))
  })

  it("accepts negative axes", {
    m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
    expect_equal(nv_quantile(m, 0.5, axes = -1L), nv_quantile(m, 0.5, axes = 2L))
    expect_equal(
      nv_quantile(m, array(c(0.25, 0.75)), axes = -1L),
      nv_quantile(m, array(c(0.25, 0.75)), axes = 2L)
    )
  })
})

describe("mean()", {
  it("errors when trim is non-zero", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_error(mean(x, trim = 0.1), "trim")
  })
  it("accepts negative dims", {
    m <- nv_matrix(as.numeric(1:6), nrow = 2)
    expect_equal(nv_mean(m, axes = -1L), nv_mean(m, axes = 2L))
    expect_equal(mean(m, axes = -1L), mean(m, axes = 2L))
  })
})

describe("nv_reverse", {
  m <- nv_matrix(1:6, nrow = 2)
  mr <- as_array(m)

  it("reverses every axis by default", {
    expect_equal(as_array(nv_reverse(m)), mr[2:1, 3:1])
    expect_equal(nv_reverse(m), nv_reverse(m, axes = c(1L, 2L)))
  })

  it("reverses the named axes only", {
    expect_equal(as_array(nv_reverse(m, axes = 1L)), mr[2:1, ])
    expect_equal(as_array(nv_reverse(m, axes = 2L)), mr[, 3:1])
  })

  it("accepts negative axes", {
    expect_equal(nv_reverse(m, axes = -1L), nv_reverse(m, axes = 2L))
  })

  it("returns the input unchanged when there is no axis to reverse", {
    expect_equal(as.vector(nv_reverse(nv_scalar(7))), 7)
    expect_equal(nv_reverse(m, axes = integer()), m)
  })

  it("agrees with rev()", {
    expect_equal(nv_reverse(m), rev(m))
  })

  it("works under jit()", {
    expect_equal(as.vector(jit(nv_reverse)(nv_array(1:3))), 3:1)
  })
})

describe("nv_argmax / nv_argmin", {
  m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
  mr <- as_array(m)

  it("reduces every axis by default, indexing the array like which.max()", {
    expect_equal(as.integer(nv_argmax(m)), which.max(mr))
    expect_equal(as.integer(nv_argmin(m)), which.min(mr))
    expect_equal(nv_argmax(m), nv_argmax(nv_flatten(m)))
    expect_shape(nv_argmax(m), integer())
  })

  it("reduces the named axes only", {
    expect_equal(nv_argmax(m, axes = 2L), prim_argmax(m, axis = 2L))
    expect_equal(nv_argmin(m, axes = 2L), prim_argmin(m, axis = 2L))
    expect_equal(as.integer(nv_argmax(m, axes = 1L)), apply(mr, 2L, which.max))
  })

  it("points at the element nv_reduce_max / nv_reduce_min returns", {
    a <- nv_array(as.numeric(c(5, 2, 9, 1, 3, 8, 4, 7, 6, 0, 2, 5)), shape = c(2L, 2L, 3L))
    ar <- as_array(a)
    for (ax in 1:3) {
      keep <- setdiff(1:3, ax)
      expect_equal(as.vector(nv_argmax(a, axes = ax)), as.vector(apply(ar, keep, which.max)))
      expect_equal(as.vector(nv_argmin(a, axes = ax)), as.vector(apply(ar, keep, which.min)))
    }
  })

  it("indexes the column-major flattening when several axes are reduced", {
    a <- nv_array(as.numeric(c(5, 2, 9, 1, 3, 8, 4, 7, 6, 0, 2, 5)), shape = c(2L, 2L, 3L))
    ar <- as_array(a)
    got <- as.integer(nv_argmax(a, axes = c(1L, 3L)))
    # for each kept position along axis 2, the reduced block `ar[, j, ]` is
    # indexed the way which.max() indexes it
    expected <- vapply(1:2, function(j) which.max(ar[, j, ]), integer(1L))
    expect_equal(got, expected)
    expect_equal(as.integer(nv_argmin(a, axes = c(2L, 3L))), apply(ar, 1L, which.min))
    expect_equal(nv_argmax(a, axes = c(1L, 2L, 3L)), nv_argmax(nv_flatten(a)))
  })

  it("keeps the reduced axes at size 1 when drop = FALSE", {
    expect_shape(nv_argmax(m, axes = 2L, drop = FALSE), c(2L, 1L))
    expect_shape(nv_argmax(m, drop = FALSE), c(1L, 1L))
    expect_equal(
      as.vector(nv_argmax(m, axes = 2L, drop = FALSE)),
      as.vector(nv_argmax(m, axes = 2L))
    )
  })

  it("returns 1 for a scalar, like which.max() does", {
    expect_equal(as.integer(nv_argmax(nv_scalar(3))), 1L)
    expect_equal(as.integer(nv_argmin(nv_scalar(3))), 1L)
  })

  it("accepts negative axes", {
    expect_equal(nv_argmax(m, axes = -1L), nv_argmax(m, axes = 2L))
    expect_equal(nv_argmin(m, axes = -1L), nv_argmin(m, axes = 2L))
  })
})

# Regression for r-xla/anvl#343: R literals must adopt the device of their
# AnvlArray siblings rather than being placed on the default device.
describe("literals adopt device of array siblings", {
  dev1 <- nv_device("cpu:1")

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

  it("nv_ifelse when only one of the three names a device", {
    # `nv_ifelse()` aligns all three arguments but promotes only the two values,
    # which is why it works below `as_anvl_arrays()`. Each direction of that is
    # load-bearing: dropping either half puts an input on the default device and
    # the call fails.
    #
    # Only `pred` names one -- the branches are built on its device rather than
    # on the default one.
    out <- nv_ifelse(nv_array(c(TRUE, FALSE), device = dev1), 1, 0)
    expect_true(eq_device(device(out), dev1))
    # Only a branch names one, and `pred` is a bare R value.
    out <- nv_ifelse(TRUE, nv_array(c(1, 2), dtype = "f64", device = dev1), 0)
    expect_true(eq_device(device(out), dev1))
  })

  it("nv_ifelse promotes only the branches, leaving pred a bool", {
    # An R branch value yields to the other branch's dtype; including `pred` in
    # the promotion would convert it out of `bool`, which prim_ifelse rejects.
    out <- nv_ifelse(arr(TRUE, FALSE), nv_array(c(1L, 2L), dtype = "i8"), 3L)
    expect_dtype(out, "i8")
    out <- nv_ifelse(arr(TRUE, FALSE), nv_array(c(1, 2), dtype = "f64"), sqrt(2))
    expect_dtype(out, "f64")
    expect_identical(as.vector(out)[[2L]], sqrt(2))
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
  it("promotes its operands instead of refusing them", {
    # The primitive underneath requires operands that already agree, and that
    # requirement used to pass straight through, so an `f32` and an `f64` were
    # refused rather than meeting at `f64` as `nv_matmul()` does.
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    b_vec <- c(1, 2)
    x <- nv_solve(nv_array(A_mat, dtype = "f32"), nv_array(b_vec, dtype = "f64"))
    expect_dtype(x, "f64")
    expect_equal(as_array(x), array(solve(A_mat, b_vec)), tolerance = 1e-5)
  })

  it("converts an integer system to the default float, like base R's solve()", {
    A_mat <- matrix(c(3L, 1L, 1L, 2L), nrow = 2)
    b_vec <- c(9L, 8L)
    x <- nv_solve(nv_array(A_mat), nv_array(b_vec))
    expect_dtype(x, default_float())
    expect_equal(as_array(x), array(solve(A_mat, b_vec)), tolerance = 1e-5)
  })

  it("lets an integer operand yield to the float it meets", {
    # Unlike numpy, where an integer operand forces the whole solve to f64.
    x <- nv_solve(
      nv_array(matrix(c(3, 1, 1, 2), nrow = 2), dtype = "f32"),
      nv_array(c(9L, 8L))
    )
    expect_dtype(x, "f32")
  })

  it("matches base R for matrix b (output stays a 2-D matrix)", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    b_mat <- matrix(c(1, 2), nrow = 2)
    x <- nv_solve(nv_array(A_mat, dtype = "f64"), nv_array(b_mat, dtype = "f64"))
    expect_shape(x, c(2L, 1L))
    expect_equal(as_array(x), solve(A_mat, b_mat), tolerance = 1e-5)
  })

  it("matches base R for vector b (output stays a 1-D vector)", {
    A_mat <- matrix(c(4, 3, 6, 3), nrow = 2)
    x <- nv_solve(nv_array(A_mat, dtype = "f64"), nv_array(c(1, 2), dtype = "f64"))
    expect_shape(x, 2L)
    expect_equal(as_array(x), array(solve(A_mat, c(1, 2))), tolerance = 1e-5)
  })
})

describe("nv_triangular_solve", {
  it("promotes its operands instead of refusing them", {
    # As in `nv_solve()`: an `f32` and an `f64` meet at `f64` now, where the
    # primitive's "operands must already agree" used to pass straight through.
    L_mat <- matrix(c(3, 1, 0, 2), nrow = 2)
    b <- c(6, 5)
    x <- nv_triangular_solve(nv_array(L_mat, dtype = "f32"), nv_array(b, dtype = "f64"))
    expect_dtype(x, "f64")
    expect_equal(as_array(x), array(solve(L_mat, b)), tolerance = 1e-5)
  })

  it("converts an integer system to the default float", {
    L_mat <- matrix(c(3L, 1L, 0L, 2L), nrow = 2)
    b <- c(6L, 5L)
    x <- nv_triangular_solve(nv_array(L_mat), nv_array(b))
    expect_dtype(x, default_float())
    expect_equal(as_array(x), array(solve(L_mat, b)), tolerance = 1e-5)
  })

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
    expect_shape(out$L, c(2L, 2L))
    expect_shape(out$U, c(2L, 2L))
    expect_shape(out$pivots, 2L)
    expect_shape(out$permutation, 2L)
    L <- as_array(out$L)
    U <- as_array(out$U)
    # `as.integer()`: the permutation follows the default integer data type, and
    # an `i64` array materializes as a `bit64::integer64`, which cannot index.
    permutation <- as.integer(as_array(out$permutation))
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
    expect_shape(out$L, c(m, k))
    expect_shape(out$U, c(k, n))
    L <- as_array(out$L)
    U <- as_array(out$U)
    # `as.integer()`: the permutation follows the default integer data type, and
    # an `i64` array materializes as a `bit64::integer64`, which cannot index.
    permutation <- as.integer(as_array(out$permutation))
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
    expect_shape(out$L, c(m, k))
    expect_shape(out$U, c(k, n))
    L <- as_array(out$L)
    U <- as_array(out$U)
    # `as.integer()`: the permutation follows the default integer data type, and
    # an `i64` array materializes as a `bit64::integer64`, which cannot index.
    permutation <- as.integer(as_array(out$permutation))
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
    expect_shape(out, c(0L, 0L))
  })
})

describe("nv_transpose", {
  it("reverses the dimensions by default", {
    x <- nv_array(array(1:24, c(2, 3, 4)))
    expect_equal(nv_transpose(x), prim_transpose(x, c(3L, 2L, 1L)))
  })
  it("accepts a negative permutation", {
    x <- nv_array(array(1:24, c(2, 3, 4)))
    expect_equal(nv_transpose(x, c(-1L, -2L, -3L)), nv_transpose(x, c(3L, 2L, 1L)))
    expect_error(nv_transpose(x, c(1L, 2L, -4L)), "between 1 and 3, or between -3 and -1")
  })
})

describe("nv_reshape", {
  it("infers a single -1 entry from the number of elements", {
    x <- nv_array(1:6)
    expect_shape(nv_reshape(x, c(2, -1)), c(2L, 3L))
    expect_shape(nv_reshape(x, c(-1, 3)), c(2L, 3L))
    expect_shape(nv_reshape(nv_array(1:12), c(2, -1, 2)), c(2L, 3L, 2L))
  })
  it("flattens with a lone -1", {
    x <- nv_array(1:6, shape = c(2, 3))
    expect_equal(nv_reshape(x, -1), nv_flatten(x))
    expect_shape(nv_reshape(nv_scalar(1), -1), 1L)
  })
  it("rejects more than one -1", {
    expect_error(nv_reshape(nv_array(1:6), c(-1, -1)), "at most one")
  })
  it("rejects a shape that does not divide evenly", {
    expect_error(nv_reshape(nv_array(1:6), c(4, -1)), "Cannot infer the size of axis")
  })
  it("rejects negative values other than -1", {
    expect_error(nv_reshape(nv_array(1:6), c(2, -2)), "must contain only non-negative")
  })
  it("agrees with base R's dim<- from rank 1 through 4", {
    shapes <- list(
      list(24L, c(2L, 3L, 4L)),
      list(c(4L, 6L), c(3L, 8L)),
      list(c(2L, 3L, 4L), c(4L, 6L)),
      list(c(2L, 1L, 3L, 4L), c(6L, 1L, 4L)),
      list(c(3L, 0L), c(0L, 2L, 3L))
    )
    for (s in shapes) {
      x <- array(as.numeric(seq_len(prod(s[[1L]]))), s[[1L]])
      expected <- x
      dim(expected) <- s[[2L]]
      expect_equal(as_array(nv_reshape(x, s[[2L]])), expected, info = shape_repr(s[[1L]]))
      expect_equal(as_array(jit(nv_reshape, static = "shape")(x, s[[2L]])), expected)
    }
  })
})

describe("nv_flatten", {
  it("flattens column-major, like as.vector()", {
    x <- matrix(1:4, nrow = 2)
    expect_equal(nv_flatten(nv_array(x)), nv_array(as.vector(x)))
    a <- array(as.numeric(1:24), c(2L, 3L, 4L))
    expect_equal(as.vector(nv_flatten(a)), as.vector(a))
    expect_equal(as.vector(jit(nv_flatten)(a)), as.vector(a))
  })
  it("works for 1D input", {
    x <- nv_array(1:3)
    expect_equal(
      nv_flatten(x),
      x
    )
  })
  it("makes a scalar a length-1 array", {
    expect_equal(nv_flatten(1L), nv_array(1L))
    expect_shape(nv_flatten(nv_scalar(1)), 1L)
  })
  it("works with empty input", {
    expect_equal(
      nv_flatten(nv_empty("f32", c(2, 0))),
      nv_empty("f32", 0L)
    )
  })
})

describe("nv_mod", {
  it("follows base R flooring semantics across sign combos", {
    lhs <- c(7, -7, 7, -7, 0, 5, -5, 1, 1.4)
    rhs <- c(3, 3, -3, -3, 3, 5, 5, -3, -0.2)
    expect_equal(
      as.vector(nv_mod(nv_array(lhs, dtype = "f64"), nv_array(rhs, dtype = "f64"))),
      lhs %% rhs,
      tolerance = 1e-12
    )
    expect_equal(as.vector(nv_mod(1L, -3L)), 1L %% -3L)
  })

  it("keeps a remainder that is tiny next to the divisor", {
    # Shifting by the divisor rounds such a remainder away, so it is only
    # applied where the sign of the truncating remainder actually differs.
    lhs <- c(1e-20, -1e-20, 1e-300)
    rhs <- c(1, 1, 1)
    expect_equal(
      as.vector(nv_mod(nv_array(lhs, dtype = "f64"), nv_array(rhs, dtype = "f64"))),
      lhs %% rhs
    )
  })

  it("keeps the operands' data type, rather than floating an integer", {
    # The shift literals used to be bare R doubles, which promoted an integer
    # remainder to a float -- and `nv_floor_div()`, which subtracts the
    # remainder, floated with it.
    x <- nv_array(c(7L, -7L, 8L, -8L))
    y <- nv_array(c(3L, 3L, -3L, -3L))
    expect_dtype(nv_mod(x, y), default_int())
    expect_dtype(nv_floor_div(x, y), default_int())
    expect_equal(as.integer(nv_mod(x, y)), as.vector(x) %% as.vector(y))
    expect_equal(as.integer(nv_floor_div(x, y)), as.vector(x) %/% as.vector(y))

    # Unsigned stays unsigned, and a float stays that float.
    u <- nv_array(7L, dtype = "ui8")
    expect_dtype(nv_mod(u, nv_array(2L, dtype = "ui8")), "ui8")
    f <- nv_array(c(7, -7), dtype = "f64")
    expect_dtype(nv_mod(f, nv_array(c(3, 3), dtype = "f64")), "f64")
  })

  it("is NaN for a zero divisor and passes NaN through, like base R", {
    lhs <- c(5, -5, 0, Inf, -Inf, NaN, 7)
    rhs <- c(0, 0, 0, 3, 3, 3, Inf)
    expect_equal(
      as.vector(nv_mod(nv_array(lhs, dtype = "f64"), nv_array(rhs, dtype = "f64"))),
      lhs %% rhs
    )
  })
})

describe("nv_scan", {
  cumsum_body <- function(carry, x) {
    s <- carry + x
    list(carry = s, out = s)
  }

  # What `prim_scan()` does with the loop is tested in
  # test-primitives-stablehlo.R; `nv_scan()` adds bare arrays in place of
  # lists, `xs = NULL` and a trip count read off `xs`.
  it("takes bare arrays and matches nv_cumsum", {
    x <- c(1, 2, 3, 4)
    res <- nv_scan(nv_scalar(0), cumsum_body, xs = nv_array(x))
    expect_equal(as.numeric(res$out), as.numeric(nv_cumsum(nv_array(x))))
    expect_equal(as.numeric(res$carry), sum(x))
  })

  it("runs fori-style with xs = NULL and an explicit length", {
    res <- nv_scan(
      init = nv_scalar(1L),
      body = function(carry, x) {
        expect_null(x)
        list(carry = carry + 1L, out = carry * 2L)
      },
      length = 3L
    )
    expect_equal(as.numeric(res$out), c(2, 4, 6))
    expect_equal(as.numeric(res$carry), 4)
  })

  it("treats an empty xs like xs = NULL", {
    res <- nv_scan(
      init = nv_scalar(1L),
      body = function(carry, x) {
        expect_null(x)
        list(carry = carry + 1L, out = carry * 2L)
      },
      xs = list(),
      length = 3L
    )
    expect_equal(as.numeric(res$out), c(2, 4, 6))
    expect_equal(as.numeric(res$carry), 4)
  })

  it("reads the trip count off xs, and demands it when there is none", {
    expect_error(nv_scan(nv_scalar(0), cumsum_body), "`length` is required")
    expect_error(nv_scan(nv_scalar(0), cumsum_body, xs = list()), "`length` is required")
    expect_error(
      nv_scan(nv_scalar(0), cumsum_body, xs = nv_scalar(1)),
      "at least one axis"
    )
  })

  # `body`, `reverse`, `length` and every leaf of `xs` are `prim_scan()`'s
  # contract; this only pins that its errors reach the caller through here.
  it("leaves the rest of the contract to prim_scan", {
    x <- nv_array(c(1, 2, 3, 4))
    expect_error(nv_scan(nv_scalar(0), "not a function", xs = x), "must be a function")
    expect_error(nv_scan(nv_scalar(0), cumsum_body, xs = x, reverse = NA), "May not be NA")
    expect_error(nv_scan(nv_scalar(0), cumsum_body, length = -1L), "not >= 0")
    expect_error(
      nv_scan(nv_scalar(0), cumsum_body, xs = x, length = 9L),
      "size 9 along axis 1, not 4"
    )
  })
})
describe("the default integer", {
  it("decides the data type of the indices an operation returns", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    local_default_dtypes(c(int = "i64"))
    i64 <- as_dtype("i64")
    expect_dtype(nv_argmax(x), i64)
    expect_dtype(nv_argmin(x), i64)
    expect_dtype(nv_argsort(x), i64)
    expect_dtype(nv_cummax(x, indices = TRUE)$indices, i64)
    expect_dtype(nv_cummin(x, indices = TRUE)$indices, i64)
    # `hlo_top_k` fixes its indices at i32, so these are converted.
    expect_dtype(nv_top_k(x, k = 2L, indices = TRUE)$indices, i64)
    # And in a trace, where the program is keyed on the defaults.
    expect_dtype(jit(function(x) nv_argmax(x))(x), i64)
    expect_dtype(jit(function(x) nv_argsort(x))(x), i64)
    expect_dtype(jit(function(x) nv_cummin(x, indices = TRUE)$indices)(x), i64)
    expect_dtype(jit(function(x) nv_top_k(x, k = 2L, indices = TRUE)$indices)(x), i64)
  })

  it("does not change the indices themselves", {
    x <- nv_array(c(3, 1, 4, 1, 5))
    at_i32 <- list(
      argmax = as_array(nv_argmax(x)),
      argsort = as_array(nv_argsort(x)),
      cummax = as_array(nv_cummax(x, indices = TRUE)$indices),
      top_k = as_array(nv_top_k(x, k = 2L, indices = TRUE)$indices)
    )
    local_default_dtypes(c(int = "i64"))
    expect_equal(as_array(nv_argmax(x)), at_i32$argmax)
    expect_equal(as_array(nv_argsort(x)), at_i32$argsort)
    expect_equal(as_array(nv_cummax(x, indices = TRUE)$indices), at_i32$cummax)
    expect_equal(as_array(nv_top_k(x, k = 2L, indices = TRUE)$indices), at_i32$top_k)
  })

  it("decides the data type of an LU decomposition's pivots", {
    # LAPACK's getrf writes 32-bit pivots, so `pivots` and `permutation` are
    # converted after the custom call rather than produced at the default.
    a <- nv_matrix(c(4, 3, 6, 3, 2, 8, 1, 5, 7), nrow = 3, dtype = "f64")
    at_i32 <- lapply(nv_lu(a)[c("pivots", "permutation")], as_array)
    local_default_dtypes(c(int = "i64"))
    factored <- nv_lu(a)
    expect_dtype(factored$pivots, "i64")
    expect_dtype(factored$permutation, "i64")
    expect_equal(as_array(factored$pivots), at_i32$pivots)
    expect_equal(as_array(factored$permutation), at_i32$permutation)
  })
})

test_that("assert_shapevec() rejects what it cannot represent", {
  expect_error(nv_fill(1, shape = 2.7), "must contain whole numbers")
  expect_error(nv_fill(1, shape = 1e10), "must contain whole numbers")
  expect_error(nv_fill(1, shape = Inf), "must contain whole numbers")
  expect_error(nv_fill(1, shape = c(-1L, 2L)), "must not contain a negative axis size")
  expect_equal(shape(nv_fill(1, shape = c(0L, 3L))), c(0L, 3L))
  expect_error(nv_fill(1, shape = c(-1L, 2L)), "`shape`")
})

test_that("a constructor that fills internally works at every data type", {
  # These fill at a data type they do not know statically, writing a plain `0`
  # or `1`, so they are what `assert_fill_value()` has to keep accepting.
  expect_equal(as.vector(nv_eye(2L, dtype = "bool")), c(TRUE, FALSE, FALSE, TRUE))
  expect_equal(dtype(nv_eye(2L, dtype = "i32")), as_dtype("i32"))
  expect_equal(as.integer(nv_diag(nv_array(c(1L, 2L)))), c(1L, 0L, 0L, 2L))
  b <- nv_array(rep(TRUE, 4L), shape = c(2L, 2L))
  expect_equal(as.vector(nv_tril(b)), c(TRUE, TRUE, FALSE, TRUE))
  expect_equal(as.vector(nv_triu(b)), c(TRUE, FALSE, TRUE, TRUE))
})
test_that("the floating-point nv_* functions refuse a boolean", {
  # `int_to_float()` used to pass a boolean through and leave the rejection to
  # the primitive, which `nv_cospi()` never reached: its `+ 1/2` promoted the
  # boolean to a float first.
  expect_error(nv_cospi(nv_array(TRUE)), "`x` must be a numeric data type")
  expect_error(nv_cospi(TRUE), "`x` must be a numeric data type")
  expect_error(nv_sinpi(nv_array(TRUE)), "`x` must be a numeric data type")
  expect_error(nv_tanpi(nv_array(TRUE)), "`x` must be a numeric data type")
  expect_error(nv_sin(nv_array(TRUE)), "`x` must be a numeric data type")
  expect_error(nv_atan2(nv_array(TRUE), nv_array(1)), "`lhs` must be a numeric data type")
  expect_error(nv_polygamma(nv_array(TRUE), nv_array(1)), "`n` must be a numeric data type")
  # A boolean meets a float at the float, so a check on the promoted operands
  # alone would let one into the linear algebra functions.
  bool_mat <- nv_array(rep(TRUE, 4L), shape = c(2L, 2L))
  rhs <- nv_array(c(1, 2), shape = c(2L, 1L), dtype = "f32")
  expect_error(nv_solve(bool_mat, rhs), "`a` must be a numeric data type")
  expect_error(nv_triangular_solve(bool_mat, rhs), "`a` must be a numeric data type")
  expect_error(nv_matmul(bool_mat, bool_mat), "`lhs` must be a numeric data type")
  expect_error(nv_crossprod(bool_mat, bool_mat), "`lhs` must be a numeric data type")
  # An integer is still accepted and converted to a float.
  expect_equal(as.vector(as_array(nv_cospi(nv_array(1L)))), -1, tolerance = 1e-6)
  expect_equal(dtype(nv_sin(nv_array(1L))), default_float())
})

test_that("nv_conv1d/2d/3d promote their operands", {
  # The `nv_*` layer promotes, as `nv_matmul()` does; `prim_convolution()`
  # underneath still requires operands that already agree.
  x32 <- nv_array(1:5, shape = c(1, 1, 5), dtype = "f32")
  x64 <- nv_array(1:5, shape = c(1, 1, 5), dtype = "f64")
  w32 <- nv_array(c(1, 0, -1), shape = c(1, 1, 3), dtype = "f32")
  w64 <- nv_array(c(1, 0, -1), shape = c(1, 1, 3), dtype = "f64")
  expect_equal(dtype(nv_conv1d(x32, w64)), as_dtype("f64"))
  expect_equal(dtype(nv_conv1d(x64, w32)), as_dtype("f64"))
  expect_equal(dtype(nv_conv1d(x32, w32)), as_dtype("f32"))
  # An integer input meets a float weight at the float; it used to be refused.
  expect_equal(
    dtype(nv_conv1d(nv_array(1:5, shape = c(1, 1, 5)), w32)),
    as_dtype("f32")
  )
  expect_equal(
    dtype(nv_conv2d(
      nv_array(1:16, shape = c(1, 1, 4, 4), dtype = "f32"),
      nv_fill(1, shape = c(1, 1, 3, 3), dtype = "f64")
    )),
    as_dtype("f64")
  )
  expect_equal(
    dtype(nv_conv3d(
      nv_array(1:18, shape = c(1, 1, 2, 3, 3), dtype = "f32"),
      nv_fill(1, shape = c(1, 1, 1, 2, 2), dtype = "f64")
    )),
    as_dtype("f64")
  )
})

test_that("nv_top_k checks `k` before coercing it", {
  # `as.integer()` first silently truncated a fractional `k` and accepted a
  # logical one, where `prim_top_k()` refuses both.
  x3 <- nv_array(c(1, 2, 3))
  expect_error(nv_top_k(x3, 1.5), "`k` must be a single whole number")
  expect_error(nv_top_k(x3, TRUE), "`k` must be a single whole number")
  expect_error(nv_top_k(x3, 10L), "`k` must be a single whole number")
  expect_error(nv_top_k(x3, 0L), "`k` must be a single whole number")
  expect_equal(as.vector(as_array(nv_top_k(x3, 2L))), c(3, 2))
})

test_that("the flag and enum arguments are checked in the nv_* layer", {
  x <- nv_array(c(1, 2, 3, 4))
  for (f in list(nv_reduce_sum, nv_reduce_prod, nv_reduce_max, nv_reduce_min, nv_mean)) {
    expect_error(f(x, nan_rm = "yes"), "logical flag")
  }
  expect_error(nv_cumsum(x, nan_rm = "yes"), "logical flag")
  expect_error(nv_cummax(x, indices = "yes"), "logical flag")
  expect_error(nv_argmax(x, nan_rm = "yes"), "logical flag")
  expect_error(nv_median(x, nan_rm = "yes"), "logical flag")
  expect_error(nv_reduce_sum(x, axes = 1L, drop = "yes"), "logical flag")
  m <- nv_array(matrix(c(4, 2, 2, 3), 2), dtype = "f32")
  expect_error(nv_chol(m, lower = "yes"), "logical flag")
  expect_error(nv_triangular_solve(m, m, lower = "yes"), "logical flag")
})

test_that("the variadic functions and `like` refuse nothing to work with", {
  expect_error(nv_concatenate(), "At least one array")
  expect_error(nv_rbind(), "At least one array")
  expect_error(nv_cbind(), "At least one array")
  expect_error(nv_broadcast_arrays(), "At least one array")
  expect_error(nv_broadcast_scalars(), "At least one array")
  expect_error(nv_promote_to_common(), "At least one array")
  expect_error(nv_fill_like(1, 2, shape = 2L), "must be an array")
})

test_that("nv_inv reports its own argument, and gradient accepts any float", {
  expect_error(nv_inv(nv_array(matrix(TRUE, 2, 2))), "`x` must be a numeric data type")
  # The check and the message agree on what "float" means.
  expect_error(
    jit(gradient(function(x) nv_reduce_sum(nv_convert(x, "i32"))))(nv_array(c(1, 2))),
    "float scalar"
  )
})

test_that("the API layer checks what its pages promise", {
  x3 <- nv_array(c(3, 1, 2))
  f23 <- nv_array(matrix(1:6, 2, 3) + 0)

  # `nv_top_k()` coerced `k` before checking it, so a fractional or logical `k`
  # was silently truncated where `prim_top_k()` refuses both -- and
  # `indices` reached a bare `if()`.
  expect_error(nv_top_k(x3, 1.5), "`k` must be a single whole number")
  expect_error(nv_top_k(x3, TRUE), "`k` must be a single whole number")
  expect_error(nv_top_k(x3, 10L), "`k` must be a single whole number")
  expect_error(nv_top_k(x3, 0L), "`k` must be a single whole number")
  expect_error(nv_top_k(x3, 1L, indices = 1), "logical flag")
  expect_equal(as.vector(as_array(nv_top_k(x3, 2L))), c(3, 2))

  # `nv_quantile()`'s bad-`probs` message was raw `checkmate` output.
  expect_error(nv_quantile(x3, 1.5), "`probs` must be probabilities")
  expect_error(nv_quantile(x3, NA_real_), "`probs` must be probabilities")

  # `nv_matmul()` left conformability to `prim_dot_general()`, which reports it
  # in terms of `contracting_axes` with 0-based numbers.
  expect_error(nv_matmul(f23, f23), "are not conformable")
  expect_error(
    nv_matmul(nv_array(array(1, c(2, 2, 2))), nv_array(matrix(1, 2, 2))),
    "same number of axes"
  )
  expect_error(
    nv_matmul(nv_array(array(1, c(2, 2, 2))), nv_array(array(1, c(3, 2, 2)))),
    "must have the same batch axes"
  )
  expect_equal(shape(nv_matmul(f23, nv_array(matrix(1:6, 3, 2) + 0))), c(2L, 2L))

  # A rank mismatch and a size mismatch are different mistakes.
  expect_error(nv_concatenate(f23, nv_array(c(1, 2, 3)), axis = 1L), "same number of axes")
  expect_error(
    nv_concatenate(f23, nv_array(matrix(1:4, 2, 2) + 0), axis = 1L),
    "same shape apart from axis 1"
  )

  # `nv_conv*` reported `kernel_input_feature_dimension` and `N`, neither of
  # which is an argument of theirs.
  expect_error(
    nv_conv1d(nv_array(array(1, c(1, 2, 4))), nv_array(array(1, c(1, 3, 2)))),
    "`weight`'s second axis"
  )
  expect_error(
    nv_conv1d(nv_array(matrix(1, 2, 2)), nv_array(matrix(1, 2, 2))),
    "must have 3 axes for a 1-D convolution"
  )
  expect_equal(
    shape(nv_conv1d(nv_array(array(1, c(1, 1, 5))), nv_array(array(1, c(1, 1, 3))))),
    c(1L, 1L, 3L)
  )
})

test_that("shape mismatches print the shapes once each", {
  a <- nv_array(matrix(1:6 / 1, 2), dtype = "f32")
  b <- nv_array(matrix(1:6 / 1, 3), dtype = "f32")
  expect_error(nv_concatenate(a, b, axis = 1L), "\\(2x3\\), \\(3x2\\)")
  expect_error(
    nv_rbind(a, nv_array(matrix(1:8 / 1, 2), dtype = "f32")),
    "\\(2x3\\), \\(2x4\\)"
  )
})

test_that("a constructor that fills internally works at every data type", {
  # These fill at a data type they do not know statically, writing a plain `0`
  # or `1`, so they are what `assert_fill_value()` has to keep accepting.
  expect_equal(as.vector(nv_eye(2L, dtype = "bool")), c(TRUE, FALSE, FALSE, TRUE))
  expect_equal(dtype(nv_eye(2L, dtype = "i32")), as_dtype("i32"))
  expect_equal(as.integer(nv_diag(nv_array(c(1L, 2L)))), c(1L, 0L, 0L, 2L))
  b <- nv_array(rep(TRUE, 4L), shape = c(2L, 2L))
  expect_equal(as.vector(nv_tril(b)), c(TRUE, TRUE, FALSE, TRUE))
  expect_equal(as.vector(nv_triu(b)), c(TRUE, FALSE, TRUE, TRUE))
})

test_that("prim_chol and nv_chol accept batched inputs", {
  # The lowering broadcasts its triangle mask over the batch axes, and both
  # pages promise batch support.
  spd <- matrix(c(4, 1, 1, 1, 4, 1, 1, 1, 4), nrow = 3)
  bx <- array(NA_real_, dim = c(2L, 3L, 3L))
  bx[1L, , ] <- spd
  bx[2L, , ] <- spd * 2
  out <- nv_chol(nv_array(bx, dtype = "f64"))
  expect_equal(shape(out), c(2L, 3L, 3L))
  got <- as_array(out)
  expect_equal(got[1L, , ], chol(spd), tolerance = 1e-8)
  expect_equal(got[2L, , ], chol(spd * 2), tolerance = 1e-8)

  # A single matrix still works, and the constraints still fire, on the last
  # two axes.
  expect_equal(shape(nv_chol(nv_array(spd, dtype = "f32"))), c(3L, 3L))
  expect_error(nv_chol(nv_array(matrix(1:6 / 1, 2), dtype = "f32")), "square in its last two axes")
  expect_error(nv_chol(nv_array(c(1, 2), dtype = "f32")), "at least 2 axes")
  expect_error(nv_chol(nv_array(spd)), NA)
  expect_error(nv_chol(nv_array(matrix(TRUE, 2, 2))), "must be a numeric data type")
})

describe("the linear algebra functions", {
  it("computes an integer input at the default float, like base R", {
    spd_mat <- matrix(c(4L, 2L, 2L, 3L), nrow = 2)
    spd <- nv_array(spd_mat)
    a_mat <- matrix(c(4L, 3L, 6L, 3L), nrow = 2)
    a <- nv_array(a_mat)
    m_mat <- matrix(1:6, nrow = 3)
    m <- nv_array(m_mat)

    expect_dtype(nv_chol(spd), default_float())
    expect_equal(as_array(nv_chol(spd)), chol(spd_mat + 0), tolerance = 1e-5)
    expect_dtype(nv_inv(a), default_float())
    expect_equal(as_array(nv_inv(a)), solve(a_mat + 0), tolerance = 1e-5)
    expect_dtype(nv_det(a), default_float())
    expect_equal(as.vector(as_array(nv_det(a))), det(a_mat + 0), tolerance = 1e-5)
    expect_dtype(nv_determinant(a)$modulus, default_float())
    expect_dtype(nv_qr(m)$Q, default_float())
    expect_dtype(nv_svd(m)$d, default_float())
    expect_equal(as.vector(as_array(nv_svd(m)$d)), svd(m_mat + 0)$d, tolerance = 1e-5)
    expect_dtype(nv_eigh(spd)$values, default_float())
    expect_dtype(nv_lu(a)$L, default_float())
    # The pivots stay indices whatever the input was.
    expect_dtype(nv_lu(a)$pivots, default_int())
  })

  it("leaves a float input at its own data type", {
    spd <- nv_array(matrix(c(4, 2, 2, 3), nrow = 2), dtype = "f64")
    expect_dtype(nv_chol(spd), "f64")
    expect_dtype(nv_inv(spd), "f64")
    expect_dtype(nv_det(spd), "f64")
    expect_dtype(nv_eigh(spd)$values, "f64")
  })

  it("still refuses a boolean input", {
    b <- nv_array(rep(TRUE, 4L), shape = c(2L, 2L))
    expect_error(nv_chol(b), "`x` must be a numeric data type")
    expect_error(nv_inv(b), "`x` must be a numeric data type")
    expect_error(nv_det(b), "`x` must be a numeric data type")
    expect_error(nv_qr(b), "`x` must be a numeric data type")
    expect_error(nv_svd(b), "`x` must be a numeric data type")
    expect_error(nv_eigh(b), "`x` must be a numeric data type")
    expect_error(nv_lu(b), "`x` must be a numeric data type")
  })
})

test_that("nv_quantile and nv_median interpolate at a float data type", {
  # `probs` used to be built at the key's data type, so at an integer one it
  # rounded to 0 and every quantile came back as the smallest element.
  expect_equal(as.vector(as_array(nv_median(nv_array(1:4)))), 2.5)
  expect_equal(as.vector(as_array(nv_median(nv_array(c(1, 2, 3, 4))))), 2.5)
  expect_equal(as.vector(as_array(nv_quantile(nv_array(1:4), 0.25))), 1.75)
  expect_equal(
    as.vector(as_array(nv_quantile(nv_array(1:4), array(c(0.25, 0.5, 0.75))))),
    c(1.75, 2.5, 3.25)
  )
  expect_equal(
    as.vector(as_array(nv_median(nv_array(c(TRUE, TRUE, FALSE, FALSE))))),
    0.5
  )
  # The result is a float whatever the interpolation mode does, and a non-float
  # input is interpolated at the default float rather than a fixed one.
  expect_equal(
    dtype(nv_quantile(nv_array(1:4), 0.5, interpolation = "lower")),
    default_float()
  )
  expect_equal(dtype(nv_median(nv_array(1:4, dtype = "i8"))), default_float())
  with_default_dtypes(c(float = "f64"), {
    expect_equal(dtype(nv_median(nv_array(1:4))), as_dtype("f64"))
  })
  expect_equal(dtype(nv_median(nv_array(c(1, 2), dtype = "f64"))), as_dtype("f64"))
  # `probs` is a scalar or a 1-D array.
  expect_error(
    nv_quantile(nv_array(c(1, 2, 3, 4)), array(c(0.25, 0.5), dim = c(1, 2))),
    "must be a length-1 numeric or a 1-D array"
  )
})

test_that("the quantile page's formula is the one the code computes", {
  # `h = 1 + (n - 1) * q` in 1-based terms, as the page now states.
  x <- nv_array(c(1, 2, 3, 4), dtype = "f64")
  n <- 4L
  sorted <- c(1, 2, 3, 4)
  for (q in c(0, 0.1, 0.3, 0.5, 0.75, 1)) {
    h <- 1 + (n - 1) * q
    lo <- floor(h)
    hi <- ceiling(h)
    frac <- h - lo
    want <- list(
      linear = (1 - frac) * sorted[lo] + frac * sorted[hi],
      lower = sorted[lo],
      higher = sorted[hi],
      nearest = if (frac < 0.5) sorted[lo] else sorted[hi],
      midpoint = (sorted[lo] + sorted[hi]) / 2
    )
    for (mode in names(want)) {
      expect_equal(
        as.vector(as_array(nv_quantile(x, q, interpolation = mode))),
        want[[mode]],
        tolerance = 1e-12,
        info = paste(mode, q)
      )
    }
  }
})

describe("nv_quantile selection fast path", {
  # probs that all lie in one half of the axis route through a top_k window
  # instead of a full sort. An array probs spanning both ends (0.1 and 0.9)
  # forces the sort path, so the two must agree exactly.
  it("matches the sort path on random data with NaNs", {
    withr::local_seed(42)
    v <- rnorm(101)
    v[sample(101, 30)] <- NaN
    x <- nv_array(v)
    for (q in c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)) {
      for (rm in c(TRUE, FALSE)) {
        sel <- as.numeric(as_array(nv_quantile(x, q, nan_rm = rm)))
        srt <- as.numeric(as_array(nv_quantile(x, array(c(0.1, 0.9, q)), nan_rm = rm)))[3L]
        expect_identical(sel, srt, info = sprintf("q = %s, nan_rm = %s", q, rm))
      }
    }
  })
  it("matches the sort path along a middle axis of a 3-D array", {
    withr::local_seed(1)
    a <- array(rnorm(7 * 55 * 6), c(7, 55, 6))
    a[sample(length(a), 500)] <- NaN
    x <- nv_array(a)
    srt <- as_array(nv_quantile(x, array(c(0.1, 0.9, 0.5, 0.8)), axes = 2L, nan_rm = TRUE))
    expect_identical(as_array(nv_median(x, axes = 2L, nan_rm = TRUE)), srt[3L, , ])
    expect_identical(as_array(nv_quantile(x, 0.8, axes = 2L, nan_rm = TRUE)), srt[4L, , ])
    # and against the R reference
    expect_equal(srt[3L, , ], apply(a, c(1, 3), median, na.rm = TRUE), tolerance = 1e-6)
    expect_equal(srt[4L, , ], apply(a, c(1, 3), quantile, 0.8, na.rm = TRUE, names = FALSE), tolerance = 1e-6)
  })
  it("handles all-NaN slices and the extremes under nan_rm", {
    x <- nv_array(matrix(c(NaN, NaN, NaN, 3, NaN, 1), 3L, 2L))
    for (q in c(0, 0.25, 0.75, 1)) {
      out <- as.numeric(as_array(nv_quantile(x, q, axes = 1L, nan_rm = TRUE)))
      expect_true(is.nan(out[1L]), info = sprintf("q = %s", q))
      expect_equal(out[2L], quantile(c(3, 1), q, names = FALSE), info = sprintf("q = %s", q))
    }
  })
  it("matches the sort path when the device index rounds past the window", {
    # The window is sized here in R doubles, but `h` is computed on device at
    # `dtype(x)`. At n = 22 and q = 1/7 that is 3 exactly in a double and
    # 3.0000002 in `f32`, so the device asks for the 5th smallest while a window
    # sized without slack holds 4 -- and the gather clamps to the 4th.
    # `"higher"` reads the upper index directly, where the clamp is visible;
    # under `"linear"` it is hidden by a `frac` of 2e-7.
    withr::local_seed(3)
    v <- runif(22)
    q <- 1 / 7
    sel <- as.numeric(as_array(nv_quantile(nv_array(v), q, interpolation = "higher")))
    srt <- as.numeric(as_array(
      nv_quantile(nv_array(v), array(c(0.1, 0.9, q)), interpolation = "higher")
    ))[3L]
    expect_identical(sel, srt)
  })
  it("matches the sort path on the high window", {
    # The high window is the device's `n_valid - floor((n_valid - 1) * probs)`
    # evaluated at the axis size, so an off-by-one shows up as a neighbouring
    # order statistic. `"higher"` reads the upper index directly, where a
    # clamped gather is visible rather than hidden behind a tiny `frac`.
    # `q = 1` is the tightest case: a window of exactly one element.
    for (n in c(9L, 22L, 56L)) {
      v <- (seq_len(n) * 37L) %% (n + 1L) + 0.5
      for (q in c(0.7, 8 / 11, 0.9, 10 / 11, 1)) {
        sel <- as.numeric(as_array(nv_quantile(nv_array(v), q, interpolation = "higher")))
        srt <- as.numeric(as_array(
          nv_quantile(nv_array(v), array(c(0.02, 0.98, q)), interpolation = "higher")
        ))[3L]
        expect_identical(sel, srt, info = sprintf("n = %d, q = %s", n, format(q)))
      }
    }
  })
  it("integer inputs still work", {
    x <- nv_array(c(5L, 1L, 9L, 3L), dtype = "i32")
    expect_equal(as.numeric(as_array(nv_quantile(x, 0.25, interpolation = "lower"))), 1)
    expect_equal(as.numeric(as_array(nv_quantile(x, 0.75, interpolation = "higher"))), 9)
  })
})
