describe("nv_subset and nv_subset_assign", {
  check <- function(shape, ...) {
    arr <- array(sample.int(prod(shape) * 10L, prod(shape)), dim = shape)

    quos <- rlang::enquos(...)
    spec <- parse_subset_specs(quos, shape)
    r_args <- lapply(seq_along(spec), \(i) {
      drop <- FALSE
      arg <- if (is_subset_full(spec[[i]])) {
        seq(shape[i])
      } else if (is_subset_range(spec[[i]])) {
        (spec[[i]]$start):(spec[[i]]$start + spec[[i]]$size - 1L)
      } else if (is_subset_index(spec[[i]])) {
        drop <- TRUE
        spec[[i]]$index
      } else if (is_subset_indices(spec[[i]])) {
        spec[[i]]$indices
      } else {
        cli_abort("Internal error")
      }
      list(arg, drop)
    })

    args <- lapply(r_args, \(a) a[[1L]])
    drop_axes <- vapply(r_args, \(a) a[[2L]], logical(1L))

    value_shape <- subset_spec_to_shape(spec)

    x <- nv_array(arr)

    # test nv_subset
    r_subset_result <- do.call(`[`, c(list(arr), args, list(drop = FALSE)))
    if (!length(value_shape)) {
      r_subset_result <- as.vector(r_subset_result)
    } else {
      dim(r_subset_result) <- value_shape
    }

    static_subset <- as_array(rlang::inject(nv_subset(x, !!!quos)))

    expect_equal(static_subset, r_subset_result)

    # test nv_subset_assign
    n_value <- max(1L, prod(value_shape))
    value <- sample.int(n_value * 10L, n_value)
    if (length(value_shape) > 0L) {
      dim(value) <- value_shape
    }
    r_assign_result <- do.call(`[<-`, c(list(arr), args, list(value = value)))

    v <- nv_array(value, shape = value_shape)

    static_assign <- as_array(rlang::inject(nv_subset_assign(x, !!!quos, value = v)))

    expect_equal(static_assign, r_assign_result)
  }

  it("1D: single element", {
    check(c(10L), 3L)
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

  it("1D: array(i) keeps the axis", {
    check(c(10L), array(3L))
  })

  it("2D: single element in both axes", {
    check(c(4L, 5L), 2L, 3L)
  })

  it("2D: range + full", {
    check(c(6L, 4L), 2:4, )
  })

  it("2D: single in first axis, range in second", {
    check(c(5L, 8L), 3L, 2:6)
  })

  it("2D: single in first axis, full second", {
    check(c(4L, 5L), 2L, )
  })

  it("2D: range in both axes", {
    check(c(6L, 8L), 2:4, 3:6)
  })

  it("2D: full first axis, range in second", {
    check(c(3L, 6L), , 2:4)
  })

  it("???", {
    check(c(2, 3, 2), 1:2, array(c(1L, 3L)), 1)
  })

  it("2D: gather in first axis, full second", {
    check(c(6L, 4L), array(c(1L, 3L, 5L)), )
  })

  it("2D: gather in both axes", {
    check(c(5L, 6L), array(c(1L, 3L, 5L)), array(c(2L, 4L)))
  })

  it("2D: gather in one axis, single in the other", {
    check(c(5L, 6L), array(c(2L, 4L)), 3L)
  })

  it("2D: single-element array preserves axis", {
    check(c(4L, 3L), array(2L), )
  })

  it("2D: trailing axis unspecified (defaults to full)", {
    check(c(4L, 3L), 2:3)
  })

  it("3D: range, single, full", {
    check(c(4L, 5L, 3L), 1:3, 2L, )
  })

  it("3D: all full (identity)", {
    check(c(3L, 4L, 2L), , , )
  })

  it("3D: gather in two axes, scalar in third", {
    check(c(4L, 5L, 6L), array(c(1L, 3L)), array(c(2L, 5L)), 1L)
  })

  it("3D: gather in first two axes, range in third", {
    check(c(4L, 5L, 6L), array(c(1L, 3L)), array(c(2L, 4L, 5L)), 2:4)
  })

  it("4D: 2 multi-index subsets", {
    check(c(3, 4, 2, 4), 1:2, array(c(1L, 2L, 4L)), , array(c(3L, 1L)))
  })

  # TODO: these tests use duplicate destination indices (`array(c(2L, 2L))`).
  # stablehlo.scatter is non-deterministic on GPU when indices collide, so the
  # equality check against R's last-wins result fails on CUDA. Improve check()
  # to verify that colliding cells hold one of the legal source values.
  it("5D: 3 multi-index subsets, no drop", {
    skip_if(is_cuda())
    check(c(3, 4, 2, 4, 3), 1:2, array(c(1L, 2L, 4L)), , array(c(3L, 1L)), array(c(2L, 2L)))
  })

  it("5D: 3 multi-index subsets, 1 drop", {
    skip_if(is_cuda())
    check(c(3, 4, 2, 4, 3), 1, array(c(1L, 2L, 4L)), , array(c(3L, 1L)), array(c(2L, 2L)))
  })

  it("5D: 3 multi-index subsets, 2 drops", {
    skip_if(is_cuda())
    check(c(3, 4, 2, 4, 3), 1, array(c(1L, 2L, 4L)), 2, array(c(3L, 1L)), array(c(2L, 2L)))
  })

  it("1D: first element (boundary)", {
    check(c(8L), 1L)
  })

  it("1D: last element (boundary)", {
    check(c(8L), 8L)
  })

  it("1D: length-1 range preserves axis", {
    check(c(10L), 3:3)
  })

  it("1D: full-length range (equivalent to full)", {
    check(c(6L), 1:6)
  })

  it("1D: gather with non-ascending indices", {
    check(c(10L), array(c(7L, 3L, 1L)))
  })

  # TODO: duplicate destination index — stablehlo.scatter is non-deterministic
  # on GPU. Improve check() to verify membership rather than last-wins equality.
  it("1D: gather with duplicate indices", {
    skip_if(is_cuda())
    check(c(6L), array(c(2L, 2L, 4L)))
  })

  it("2D: axis of size 1", {
    check(c(1L, 5L), 1L, 2:4)
  })

  it("3D: all axes dropped (scalar result)", {
    check(c(4L, 5L, 3L), 2L, 3L, 1L)
  })

  it("3D: trailing axes unspecified with drop", {
    check(c(4L, 5L, 3L), 2L)
  })

  # TODO: duplicate destination indices — stablehlo.scatter is non-deterministic
  # on GPU. Improve check() to verify membership rather than last-wins equality.
  it("2D: gather with duplicates in both axes", {
    skip_if(is_cuda())
    check(c(4L, 5L), array(c(1L, 1L, 3L)), array(c(2L, 2L)))
  })

  it("2D: boundary indices in both axes", {
    check(c(3L, 4L), 1:3, 1:4)
  })

  # Minimal CUDA-safe duplicate-index test: writes 100 values to position 1,
  # then checks that element 1 holds one of those values (rather than asserting
  # which specific one — that would be non-deterministic on GPU).
  it("scatter with all-colliding destination indices yields a valid write", {
    x <- nv_array(1:100)
    result <- as_array(nv_subset_assign(x, array(rep(1L, 100)), value = nv_array(101:200)))
    expect_true(as.vector(result)[1L] %in% 101:200)
    expect_equal(as.vector(result)[-1L], 2:100)
  })

  it("subset errors on R vector of length > 1", {
    x <- nv_array(1:10)
    expect_error(
      x[c(1, 2)],
      "Vectors of length > 1 are not allowed"
    )
  })

  it("errors on too many subset specs (trailing empties)", {
    x <- nv_array(1:10) # 1-D
    expect_error(x[1:5, ], "Too many subset specifications")
    expect_error(x[1:5, , ], "Too many subset specifications")
    expect_error(x[1:5, 1:3, ], "Too many subset specifications")

    y <- nv_matrix(1:6, nrow = 2) # 2-D
    expect_error(y[1, 1, ], "Too many subset specifications")
    expect_error(y[1, 1, , ], "Too many subset specifications")
  })

  it("subset_assign errors on too many subset specs (trailing empties)", {
    x <- nv_array(1:10) # 1-D
    expect_error(
      {
        x[1:5, ] <- 0L
      },
      "Too many subset specifications"
    )
    expect_error(
      {
        x[1:5, , ] <- 0L
      },
      "Too many subset specifications"
    )

    y <- nv_matrix(1:6, nrow = 2) # 2-D
    expect_error(
      {
        y[1, 1, ] <- 0L
      },
      "Too many subset specifications"
    )
  })

  it("subset_assign broadcasts scalar rhs", {
    expect_equal(
      {
        x <- nv_array(1:3)
        x[1:3] <- -1L
        x
      },
      nv_array(c(-1L, -1L, -1L))
    )
  })

  it("subset_assign errors when update shape doesn't match", {
    expect_error(
      {
        x <- nv_matrix(1:6, nrow = 2)
        x[1:2, 1:2] <- nv_array(1:2)
        x
      },
      "Update shape does not match subset shape"
    )
  })

  it("errors on out-of-bounds range (start < 1)", {
    x <- nv_array(1:10)
    expect_error(x[0:5], "out of bounds")
  })

  it("errors on out-of-bounds range (end > axis_size)", {
    x <- nv_array(1:10)
    expect_error(x[5:11], "out of bounds")
  })

  it("errors on out-of-bounds single index (< 1)", {
    x <- nv_array(1:10)
    expect_error(x[0L], "out of bounds")
  })

  it("errors on out-of-bounds single index (> axis_size)", {
    x <- nv_array(1:10)
    expect_error(x[11L], "out of bounds")
  })

  it("errors on out-of-bounds array() index", {
    x <- nv_array(1:10)
    expect_error(x[array(c(1, 11))], "out of bounds")
    expect_error(x[array(c(0, 5))], "out of bounds")
  })

  it("works with all-static indices via [", {
    r_arr <- array(1:24, dim = c(2, 3, 4))
    x <- nv_array(r_arr)
    result <- x[2L, 1L, 3L]
    expect_equal(as_array(result), r_arr[2, 1, 3])

    result2 <- x[array(c(1L, 1L)), array(c(2L, 3L)), array(c(2L, 1L))]
    expect_equal(as_array(result2), r_arr[c(1, 1), c(2, 3), c(2, 1)])
  })

  it("works with nv_arrays just like with R indices", {
    x <- {
      x <- nv_array(1:24, shape = c(2, 3, 4))
      x1 <- x[1:2, array(c(1L, 3L)), 1]
      x2 <- x[nv_array(c(1L, 2L)), array(c(1L, 3L)), nv_scalar(1L)]
      list(x1, x2)
    }
    expect_equal(x[[1]], x[[2]])

    y <- {
      x <- nv_array(1:24, shape = c(2, 3, 4))
      x1 <- x
      x2 <- x
      update <- nv_array(1:4, shape = c(2, 2))
      x1[1:2, array(c(1L, 3L)), 1] <- update
      x2[nv_array(c(1L, 2L)), array(c(1L, 3L)), nv_scalar(1L)] <- update
      list(x1, x2)
    }
    expect_equal(y[[1]], y[[2]])
  })

  it("works with nv_seq", {
    x <- {
      x <- nv_array(1:10)
      x[nv_seq(2, 5)]
    }
    expect_equal(x, nv_array(2:5))
  })
})

describe("subset_specs_start_indices", {
  it("returns all 1s for SubsetFull specs", {
    subsets <- list(SubsetFull(5L), SubsetFull(3L))
    result <- subset_specs_start_indices(subsets)
    expect_equal(dtype(result), as_dtype("i32"))
    expect_equal(as.integer(result), c(1L, 1L))
  })

  it("returns start values for SubsetRange specs", {
    subsets <- list(SubsetRange(3L, 5L), SubsetRange(2L, 4L))
    result <- subset_specs_start_indices(subsets)
    expect_equal(dtype(result), as_dtype("i32"))
    expect_equal(as.integer(result), c(3L, 2L))
  })

  it("returns the index for scalar SubsetIndices", {
    subsets <- list(SubsetIndex(nv_scalar(4L, dtype = "i64")))
    result <- subset_specs_start_indices(subsets)
    expect_equal(dtype(result), as_dtype("i64"))
    expect_equal(as.integer(result), 4L)
  })

  it("handles mixed spec types", {
    subsets <- list(
      SubsetRange(2L, 5L),
      SubsetFull(10L),
      SubsetIndex(nv_scalar(3L, dtype = "i64"))
    )
    result <- subset_specs_start_indices(subsets)
    expect_equal(dtype(result), as_dtype("i64"))
    expect_equal(as.integer(result), c(2L, 1L, 3L))
  })

  it("uses i32 for all-static subsets", {
    subsets <- list(SubsetFull(5L), SubsetRange(3L, 5L))
    result <- subset_specs_start_indices(subsets)
    expect_equal(dtype(result), as_dtype("i32"))
    expect_equal(as.integer(result), c(1L, 3L))
  })

  it("uses max integer type when dynamic indices are present", {
    subsets <- list(
      SubsetIndex(nv_scalar(2L, dtype = "i16")),
      SubsetRange(3L, 5L)
    )
    result <- subset_specs_start_indices(subsets)
    expect_equal(dtype(result), as_dtype("i32"))
  })

  it("works with a single axis", {
    subsets <- list(SubsetRange(2L, 7L))
    result <- subset_specs_start_indices(subsets)
    expect_equal(dtype(result), as_dtype("i32"))
    expect_equal(as.integer(result), 2L)
  })

  it("works with empty subset", {
    x <- {
      nv_array(1:10)[]
    }
    expect_equal(x, nv_array(1:10))
    x <- {
      x <- nv_array(1:10)
      x[] <- nv_array(2:11)
    }
    expect_equal(x, nv_array(2:11))
  })
})

test_that("nv_subset still selects with an integer index vector", {
  # Regression guard for the index data type checks the slicing primitives gained.
  x <- nv_array(c(1, 2, 3, 4, 5), dtype = "f64")
  expect_equal(as.numeric(nv_subset(x, 2:3)), c(2, 3))
})
