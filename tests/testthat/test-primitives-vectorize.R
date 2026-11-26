# Each primitive with a vectorize rule is checked by `autotest_vectorize()`
# (helper-vectorize.R): mapped over every subset of its operands, it must agree
# with applying the primitive to one slice at a time. A primitive tagged
# "elementwise", "reduction" or "cumulative" gets the rule of its tag, so one
# primitive of each tag is checked, the element-wise one also with a scalar
# operand.
#
# A rule that does more than append the batch axis is also checked on operands
# of more axes, which give `autotest_vectorize()` more batch axes to draw from
# and so check the rule's axis bookkeeping.

describe("the rule of the element-wise primitives", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(prim_add, lhs = rand_array(c(2L, 3L)), rhs = rand_array(c(2L, 3L)))
  })

  it("broadcasts a scalar operand that is mapped over, and leaves one that is not", {
    withr::local_seed(1L)
    autotest_vectorize(
      prim_clamp,
      x = rand_array(3L),
      min = rand_array(c(2L, 3L), sample = \(n) stats::runif(n, -1, 0)),
      max = rand_array(3L, sample = \(n) stats::runif(n, 0, 1))
    )
  })

  it("maps over the test of prim_ifelse() without its branches", {
    withr::local_seed(1L)
    autotest_vectorize(
      prim_ifelse,
      test = rand_array(c(2L, 3L), dtype = "bool", sample = \(n) stats::runif(n) < 0.5),
      yes = rand_array(c(2L, 3L)),
      no = rand_array(c(2L, 3L))
    )
  })
})

describe("prim_bitcast_convert", {
  it("vectorizes between data types of one width", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_bitcast_convert(x, "i32"), x = rand_array(c(2L, 3L)))
  })

  it("vectorizes to a narrower data type", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_bitcast_convert(x, "ui8"), x = rand_array(c(2L, 3L)))
    autotest_vectorize(function(x) prim_bitcast_convert(x, "ui8"), x = rand_array(c(2L, 4L, 3L)))
    autotest_vectorize(function(x) prim_bitcast_convert(x, "i16"), x = rand_array(3L))
  })

  it("vectorizes to a wider data type", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x) prim_bitcast_convert(x, "f32"),
      x = rand_array(c(4L, 2L, 3L), dtype = "ui8", sample = \(n) sample(0:255, n, replace = TRUE))
    )
  })
})

describe("the rule of the reductions", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_sum(x, axes = 2L, drop = TRUE), x = rand_array(c(2L, 4L, 3L)))
    autotest_vectorize(function(x) prim_sum(x, axes = 1:2, drop = FALSE), x = rand_array(c(2L, 4L, 3L)))
  })

  it("vectorizes a reduction to the index of an extreme", {
    withr::local_seed(1L)
    for (drop in c(TRUE, FALSE)) {
      autotest_vectorize(function(x) prim_which_min(x, axis = 2L, drop = drop), x = rand_array(c(2L, 4L, 3L)))
    }
  })
})

describe("the rule of the cumulative primitives", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_cumsum(x, axis = 2L), x = rand_array(c(2L, 4L, 3L)))
  })

  it("vectorizes a primitive with two outputs", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_cummax(x, axis = 2L), x = rand_array(c(2L, 4L, 3L)))
  })
})

describe("prim_rev", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_rev(x, axes = 2L), x = rand_array(c(2L, 4L, 3L)))
  })
})

describe("prim_transpose", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_transpose(x, perm = c(2L, 1L)), x = rand_array(c(2L, 4L, 3L)))
  })
})

describe("prim_reshape", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_reshape(x, shape = c(4L, 2L)), x = rand_array(c(2L, 4L, 3L)))
  })
})

describe("prim_broadcast_in_axes", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x) prim_broadcast_in_axes(x, shape = c(2L, 5L), broadcast_axes = 1L),
      x = rand_array(c(2L, 3L))
    )
  })

  it("vectorizes an operand of more axes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x) prim_broadcast_in_axes(x, shape = c(5L, 2L, 4L), broadcast_axes = c(2L, 3L)),
      x = rand_array(c(2L, 4L, 3L))
    )
  })
})

describe("prim_concatenate", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, y) prim_concatenate(x, y, axis = 1L),
      x = rand_array(c(2L, 2L, 3L)),
      y = rand_array(c(1L, 2L, 3L))
    )
  })

  it("vectorizes along an axis other than the first", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, y) prim_concatenate(x, y, axis = 2L),
      x = rand_array(c(2L, 4L, 3L)),
      y = rand_array(c(2L, 1L, 3L))
    )
  })
})

describe("prim_static_slice", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x) prim_static_slice(x, start_indices = c(1L, 2L), end_indices = c(2L, 4L), strides = c(1L, 2L)),
      x = rand_array(c(2L, 4L, 3L))
    )
  })
})

describe("prim_pad", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x) prim_pad(x, 0, c(1L, 0L), c(0L, 1L), c(1L, 0L)),
      x = rand_array(c(2L, 4L, 3L))
    )
  })

  it("refuses to map over the padding value", {
    f <- function(x, value) prim_pad(x, value, c(1L, 0L), c(0L, 1L), c(0L, 0L))
    expect_error(
      jit(vectorize(f, in_axes = list(value = 1L)))(rand_array(c(2L, 4L)), rand_array(3L)),
      "cannot map over operand 2 of `prim_pad\\(\\)`"
    )
  })
})

describe("prim_dynamic_slice", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x) prim_dynamic_slice(x, 2L, nv_scalar(1L, dtype = "i32"), slice_sizes = c(1L, 2L)),
      x = rand_array(c(2L, 4L, 3L))
    )
  })

  it("refuses to map over a start index", {
    f <- function(x, i) prim_dynamic_slice(x, i, 1L, slice_sizes = c(1L, 2L))
    expect_error(
      jit(vectorize(f, in_axes = list(i = 1L)))(rand_array(c(2L, 4L)), nv_array(1:3, dtype = "i32")),
      "start indices of `prim_dynamic_slice\\(\\)`"
    )
  })
})

describe("prim_gather", {
  it("vectorizes gathering rows of a matrix", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, i) {
        prim_gather(
          x,
          i,
          slice_sizes = c(1L, 3L),
          offset_axes = 2L,
          collapsed_slice_axes = 1L,
          x_batching_axes = integer(),
          start_indices_batching_axes = integer(),
          start_index_map = 1L,
          index_vector_axis = 2L
        )
      },
      x = rand_array(c(5L, 3L, 4L)),
      i = rand_indices(c(6L, 1L, 4L), 5L)
    )
  })

  it("vectorizes a gather whose index vector axis is implicit", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, i) {
        prim_gather(
          x,
          i,
          slice_sizes = 2L,
          offset_axes = 2L,
          collapsed_slice_axes = integer(),
          x_batching_axes = integer(),
          start_indices_batching_axes = integer(),
          start_index_map = 1L,
          index_vector_axis = 2L
        )
      },
      x = rand_array(c(5L, 3L)),
      i = rand_indices(c(4L, 3L), 4L)
    )
  })

  it("vectorizes a gather with batching axes of its own", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, i) {
        prim_gather(
          x,
          i,
          slice_sizes = c(1L, 1L, 3L),
          offset_axes = 3L,
          collapsed_slice_axes = 2L,
          x_batching_axes = 1L,
          start_indices_batching_axes = 1L,
          start_index_map = 2L,
          index_vector_axis = 3L
        )
      },
      x = rand_array(c(2L, 5L, 3L, 4L)),
      i = rand_indices(c(2L, 6L, 4L), 5L)
    )
  })
})

describe("prim_scatter", {
  # scatters the elements of `update` into the vector `x` at the indices in
  # the rows of the matrix `i`
  scatter_elements <- function(x, i, update, ...) {
    prim_scatter(
      x,
      i,
      update,
      update_window_axes = integer(),
      inserted_window_axes = 1L,
      x_batching_axes = integer(),
      scatter_indices_batching_axes = integer(),
      scatter_axes_to_x_axes = 1L,
      index_vector_axis = 2L,
      ...
    )
  }
  add <- function(old, new) old + new

  it("vectorizes a scatter of single elements", {
    withr::local_seed(1L)
    # distinct indices within a slice, so that which update wins is defined
    i <- nv_array(as.integer(replicate(3L, sample.int(5L, 2L))), shape = c(2L, 1L, 3L), dtype = "i32")
    autotest_vectorize(scatter_elements, x = rand_array(c(5L, 3L)), i = i, update = rand_array(c(2L, 3L)))
  })

  it("vectorizes a scatter with an update function", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, i, update) scatter_elements(x, i, update, update_fn = add),
      x = rand_array(c(5L, 3L)),
      i = rand_indices(c(4L, 1L, 3L), 5L),
      update = rand_array(c(4L, 3L))
    )
  })

  it("vectorizes a scatter whose index vector axis is implicit", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, i, update) scatter_elements(x, i, update, update_fn = add),
      x = rand_array(c(5L, 3L)),
      i = rand_indices(c(4L, 3L), 5L),
      update = rand_array(c(4L, 3L))
    )
  })

  it("vectorizes a scatter of windows", {
    withr::local_seed(1L)
    scatter_windows <- function(x, i, update) {
      prim_scatter(
        x,
        i,
        update,
        update_window_axes = seq_len(naxes(x) - 1L) + 1L,
        inserted_window_axes = 1L,
        x_batching_axes = integer(),
        scatter_indices_batching_axes = integer(),
        scatter_axes_to_x_axes = 1L,
        index_vector_axis = 2L,
        update_fn = add
      )
    }
    i <- rand_indices(c(2L, 1L, 3L), 4L)
    autotest_vectorize(scatter_windows, x = rand_array(c(4L, 3L, 3L)), i = i, update = rand_array(c(2L, 3L, 3L)))
    autotest_vectorize(
      scatter_windows,
      x = rand_array(c(4L, 3L, 2L, 3L)),
      i = i,
      update = rand_array(c(2L, 3L, 2L, 3L))
    )
  })

  it("vectorizes a scatter with batching axes of its own", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, i, update) {
        prim_scatter(
          x,
          i,
          update,
          update_window_axes = integer(),
          inserted_window_axes = 2L,
          x_batching_axes = 1L,
          scatter_indices_batching_axes = 1L,
          scatter_axes_to_x_axes = 2L,
          index_vector_axis = 3L,
          update_fn = add
        )
      },
      x = rand_array(c(2L, 5L, 3L)),
      i = rand_indices(c(2L, 4L, 1L, 3L), 5L),
      update = rand_array(c(2L, 4L, 3L))
    )
  })

  it("keeps the hints of indices it does not map over, and the uniqueness of ones it does", {
    f <- function(x, i, update) scatter_elements(x, i, update, indices_are_sorted = TRUE, unique_indices = TRUE)
    # the parameters of the scatter in the trace of `f` vectorized over `in_axes`
    params_of <- function(i, in_axes) {
      args <- list(x = nv_aval("f32", c(5L, 3L)), i = i, update = nv_aval("f32", c(2L, 3L)))
      graph <- trace_fn(vectorize(f, in_axes = in_axes), args)
      Find(\(s) s$primitive$name == "scatter", graph$statements)$params
    }
    p <- params_of(nv_aval("i32", c(2L, 1L)), list(x = -1L, update = -1L))
    expect_true(p$indices_are_sorted)
    expect_true(p$unique_indices)
    p <- params_of(nv_aval("i32", c(2L, 1L, 3L)), list(x = -1L, i = -1L, update = -1L))
    expect_false(p$indices_are_sorted)
    expect_true(p$unique_indices)
  })
})

describe("prim_dynamic_update_slice", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, update) prim_dynamic_update_slice(x, update, 1L, 2L),
      x = rand_array(c(2L, 4L, 3L)),
      update = rand_array(c(1L, 2L, 3L))
    )
  })

  it("refuses to map over a start index", {
    f <- function(x, i) prim_dynamic_update_slice(x, nv_array(c(1, 2), shape = c(1L, 2L), dtype = "f32"), i, 1L)
    expect_error(
      jit(vectorize(f, in_axes = list(i = 1L)))(rand_array(c(2L, 4L)), nv_array(1:3, dtype = "i32")),
      "start indices of `prim_dynamic_update_slice\\(\\)`"
    )
  })
})

describe("prim_sort", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, y) prim_sort(list(x, y), axis = 2L),
      x = rand_array(c(2L, 4L, 3L)),
      y = rand_array(c(2L, 4L, 3L))
    )
  })

  it("vectorizes along the first axis", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, y) prim_sort(list(x, y), axis = 1L),
      x = rand_array(c(2L, 4L, 3L)),
      y = rand_array(c(2L, 4L, 3L))
    )
  })
})

describe("prim_top_k", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_top_k(x, k = 2L, indices = TRUE), x = rand_array(c(2L, 4L, 3L)))
  })
})

describe("prim_dot_general", {
  it("vectorizes a matrix-vector product", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(lhs, rhs) {
        prim_dot_general(lhs, rhs, contracting_axes = list(2L, 1L), batching_axes = list(integer(), integer()))
      },
      lhs = rand_array(c(2L, 4L, 3L)),
      rhs = rand_array(c(4L, 3L))
    )
  })

  it("vectorizes a product with batching axes of its own", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(lhs, rhs) {
        prim_dot_general(lhs, rhs, contracting_axes = list(3L, 2L), batching_axes = list(1L, 1L))
      },
      lhs = rand_array(c(2L, 4L, 5L, 3L)),
      rhs = rand_array(c(2L, 5L, 6L, 3L))
    )
  })

  it("vectorizes products of operands of more axes", {
    withr::local_seed(1L)
    dot_general <- function(contracting, batching) {
      function(lhs, rhs) prim_dot_general(lhs, rhs, contracting_axes = contracting, batching_axes = batching)
    }
    g <- dot_general(list(3L, 1L), list(integer(), integer()))
    autotest_vectorize(g, lhs = rand_array(c(2L, 3L, 4L, 3L)), rhs = rand_array(c(4L, 5L, 3L)))
    h <- dot_general(list(c(2L, 4L), c(3L, 1L)), list(1L, 2L))
    autotest_vectorize(
      h,
      lhs = rand_array(c(2L, 4L, 5L, 6L, 3L)),
      rhs = rand_array(c(6L, 2L, 4L, 3L, 3L))
    )
  })
})

describe("prim_print", {
  it("vectorizes", {
    withr::local_seed(1L)
    capture.output(autotest_vectorize(prim_print, x = rand_array(c(2L, 3L))))
  })
})

describe("prim_sample_bits", {
  it("vectorizes into a draw for every slice", {
    withr::local_seed(1L)
    local_nv_seed(1L)
    x <- rand_array(c(2L, 3L))
    for (dtype in c("ui8", "ui32", "ui64")) {
      autotest_vectorize(function(x) list(x, prim_sample_bits(c(2L, 3L), dtype)), x = x)
    }
  })
})

describe("prim_optimization_barrier", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, y) prim_optimization_barrier(list(x, y)),
      x = rand_array(c(2L, 3L)),
      y = rand_array(c(4L, 3L))
    )
  })

  it("vectorizes the draws of nv_rnorm(), which go through it", {
    withr::local_seed(1L)
    local_nv_seed(1L)
    # Box-Muller draws in pairs, so a scalar or an odd number of draws takes
    # part of one
    for (shape in list(integer(), 1L, 2L, 3L)) {
      autotest_vectorize(
        function(x) x + nv_rnorm(shape = shape),
        x = rand_array(c(if (length(shape)) shape else 2L, 3L))
      )
    }
  })
})

# A batch of `size` symmetric positive definite matrices, of shape
# (n, n, size).
rand_spd <- function(n, size) {
  x <- array(stats::rnorm(n * n * size), c(n, n, size))
  for (i in seq_len(size)) {
    x[,, i] <- crossprod(x[,, i]) + diag(n)
  }
  nv_array(x, dtype = "f64")
}

describe("prim_chol", {
  it("vectorizes", {
    withr::local_seed(1L)
    for (lower in c(TRUE, FALSE)) {
      autotest_vectorize(function(x) prim_chol(x, lower = lower), x = rand_spd(3L, 4L))
    }
  })
})

describe("prim_triangular_solve", {
  # A batch of triangular matrices, whose diagonal keeps them well conditioned.
  rand_triangular <- function(n, size) {
    x <- array(stats::rnorm(n * n * size), c(n, n, size))
    for (i in seq_len(size)) {
      diag(x[,, i]) <- 3
    }
    nv_array(x, dtype = "f64")
  }

  it("vectorizes a solve from the left", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(a, b) {
        prim_triangular_solve(a, b, left_side = TRUE, lower = TRUE, unit_diagonal = FALSE, transpose_a = FALSE)
      },
      a = rand_triangular(3L, 4L),
      b = rand_array(c(3L, 2L, 4L), dtype = "f64")
    )
  })

  it("vectorizes a transposed solve from the right", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(a, b) {
        prim_triangular_solve(a, b, left_side = FALSE, lower = FALSE, unit_diagonal = TRUE, transpose_a = TRUE)
      },
      a = rand_triangular(3L, 4L),
      b = rand_array(c(2L, 3L, 4L), dtype = "f64")
    )
  })
})

describe("prim_qr", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(prim_qr, x = rand_array(c(4L, 3L, 5L), dtype = "f64"))
    autotest_vectorize(prim_qr, x = rand_array(c(3L, 4L, 5L), dtype = "f64"))
  })
})

describe("prim_lu", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(prim_lu, x = rand_array(c(4L, 4L, 5L), dtype = "f64"))
    autotest_vectorize(prim_lu, x = rand_array(c(3L, 4L, 5L), dtype = "f64"))
  })
})

describe("prim_svd", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(prim_svd, x = rand_array(c(4L, 3L, 5L), dtype = "f64"))
    autotest_vectorize(prim_svd, x = rand_array(c(3L, 4L, 5L), dtype = "f64"))
  })
})

describe("prim_eigh", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(prim_eigh, x = rand_spd(3L, 5L))
  })
})

# Every primitive with a vectorize rule must have been applied by one of the
# `autotest_vectorize()` calls above (see helper-vectorize.R), except one that
# gets the rule of its tag, of which one per tag suffices.
describe("vectorize rules", {
  it("are tested for every primitive with a rule of its own, and once per tag", {
    nms <- names(asNamespace("anvl"))
    prims <- Filter(\(p) inherits(p, "AnvlPrimitive"), mget(nms[startsWith(nms, "prim_")], asNamespace("anvl")))
    with_rule <- sub("^prim_", "", names(Filter(\(p) !is.null(p[["vectorize"]]), prims)))
    tested <- names(vectorize_tested)
    tags <- c("elementwise", "reduction", "cumulative")
    tagged <- lapply(stats::setNames(nm = tags), \(tag) sub("^prim_", "", names(primitives_tagged(tag))))
    expect_identical(setdiff(with_rule, c(tested, unlist(tagged))), character())
    for (tag in tags) {
      expect_true(any(tagged[[tag]] %in% tested), label = sprintf("a primitive tagged %s is tested", tag))
    }
  })
})
