# Each primitive with a vectorize rule is checked by `autotest_vectorize()`
# (helper-vectorize.R): mapped over every subset of its operands, it must agree
# with applying the primitive to one slice at a time.

positive <- function(n) stats::runif(n, 0.5, 2)
unit <- function(n) stats::runif(n, -0.9, 0.9)
small_ints <- function(n) sample(0:5, n, replace = TRUE)
coin <- function(n) stats::runif(n) > 0.5

describe("element-wise unary primitives", {
  it("vectorizes the ones on the real line", {
    withr::local_seed(1L)
    for (prim in list(
      prim_negate,
      prim_abs,
      prim_tanh,
      prim_tan,
      prim_sin,
      prim_cos,
      prim_floor,
      prim_ceiling,
      prim_sign,
      prim_exp,
      prim_expm1,
      prim_cbrt,
      prim_plogis,
      prim_asinh,
      prim_atan,
      prim_cosh,
      prim_sinh,
      prim_erf,
      prim_erfc,
      prim_is_finite,
      prim_round
    )) {
      autotest_vectorize(prim, x = rand_array(c(3L, 2L)))
    }
  })

  it("vectorizes the ones on a restricted domain", {
    withr::local_seed(1L)
    for (prim in list(prim_sqrt, prim_rsqrt, prim_log, prim_log1p, prim_digamma, prim_lgamma)) {
      autotest_vectorize(prim, x = rand_array(c(3L, 2L), sample = positive))
    }
    for (prim in list(prim_acos, prim_asin, prim_atanh, prim_erf_inv)) {
      autotest_vectorize(prim, x = rand_array(c(3L, 2L), sample = unit))
    }
    autotest_vectorize(prim_acosh, x = rand_array(c(3L, 2L), sample = \(n) stats::runif(n, 1.5, 3)))
  })

  it("vectorizes the ones on integers and booleans", {
    withr::local_seed(1L)
    autotest_vectorize(prim_not, x = rand_array(c(3L, 2L), dtype = "bool", sample = coin))
    autotest_vectorize(prim_popcnt, x = rand_array(c(3L, 2L), dtype = "i32", sample = small_ints))
  })
})

describe("element-wise binary primitives", {
  it("vectorizes the arithmetic ones", {
    withr::local_seed(1L)
    for (prim in list(prim_add, prim_mul, prim_sub, prim_pmax, prim_pmin)) {
      autotest_vectorize(prim, lhs = rand_array(c(3L, 2L)), rhs = rand_array(c(3L, 2L)))
    }
    autotest_vectorize(prim_div, lhs = rand_array(c(3L, 2L)), rhs = rand_array(c(3L, 2L), sample = positive))
    autotest_vectorize(prim_pow, x = rand_array(c(3L, 2L), sample = positive), y = rand_array(c(3L, 2L)))
    autotest_vectorize(prim_remainder, x = rand_array(c(3L, 2L)), y = rand_array(c(3L, 2L), sample = positive))
    autotest_vectorize(prim_atan2, y = rand_array(c(3L, 2L)), x = rand_array(c(3L, 2L)))
  })

  it("vectorizes the comparisons", {
    withr::local_seed(1L)
    for (prim in list(prim_eq, prim_ne, prim_gt, prim_ge, prim_lt, prim_le)) {
      autotest_vectorize(
        prim,
        lhs = rand_array(c(3L, 4L), dtype = "i32", sample = small_ints),
        rhs = rand_array(c(3L, 4L), dtype = "i32", sample = small_ints)
      )
    }
  })

  it("vectorizes the logical and bitwise ones", {
    withr::local_seed(1L)
    for (prim in list(prim_and, prim_or)) {
      autotest_vectorize(
        prim,
        lhs = rand_array(c(3L, 2L), dtype = "bool", sample = coin),
        rhs = rand_array(c(3L, 2L), dtype = "bool", sample = coin)
      )
    }
    autotest_vectorize(
      prim_xor,
      x = rand_array(c(3L, 2L), dtype = "i32", sample = small_ints),
      y = rand_array(c(3L, 2L), dtype = "i32", sample = small_ints)
    )
    for (prim in list(prim_shift_left, prim_shift_right_logical, prim_shift_right_arithmetic)) {
      autotest_vectorize(
        prim,
        x = rand_array(c(3L, 2L), dtype = "i32", sample = small_ints),
        shift = rand_array(c(3L, 2L), dtype = "i32", sample = small_ints)
      )
    }
  })
})

describe("prim_psigamma", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      prim_psigamma,
      x = rand_array(c(3L, 2L), sample = positive),
      deriv = rand_array(c(3L, 2L), sample = \(n) rep(1, n))
    )
  })
})

describe("prim_clamp", {
  it("vectorizes with bounds of the shape of x", {
    withr::local_seed(1L)
    autotest_vectorize(
      prim_clamp,
      x = rand_array(c(3L, 2L)),
      min = rand_array(c(3L, 2L), sample = \(n) stats::runif(n, -1, 0)),
      max = rand_array(c(3L, 2L), sample = \(n) stats::runif(n, 0, 1))
    )
  })

  it("vectorizes with scalar bounds", {
    withr::local_seed(1L)
    autotest_vectorize(
      prim_clamp,
      x = rand_array(c(3L, 2L)),
      min = rand_array(3L, sample = \(n) stats::runif(n, -1, 0)),
      max = rand_array(3L, sample = \(n) stats::runif(n, 0, 1))
    )
  })
})

describe("prim_ifelse", {
  it("vectorizes with an element-wise test", {
    withr::local_seed(1L)
    autotest_vectorize(
      prim_ifelse,
      test = rand_array(c(3L, 2L), dtype = "bool", sample = coin),
      yes = rand_array(c(3L, 2L)),
      no = rand_array(c(3L, 2L))
    )
  })

  it("vectorizes with a scalar test", {
    withr::local_seed(1L)
    autotest_vectorize(
      prim_ifelse,
      test = rand_array(3L, dtype = "bool", sample = coin),
      yes = rand_array(c(3L, 2L)),
      no = rand_array(c(3L, 2L))
    )
  })
})

describe("prim_convert", {
  it("vectorizes", {
    autotest_vectorize(function(x) prim_convert(x, "f64"), x = rand_array(c(3L, 2L)))
  })
})

describe("prim_bitcast_convert", {
  it("vectorizes", {
    autotest_vectorize(function(x) prim_bitcast_convert(x, "i32"), x = rand_array(c(3L, 2L)))
  })
})

describe("reductions", {
  it("vectorizes prim_sum, prim_prod, prim_max and prim_min", {
    withr::local_seed(1L)
    for (prim in list(prim_sum, prim_prod, prim_max, prim_min)) {
      autotest_vectorize(function(x) prim(x, axes = 2L, drop = TRUE), x = rand_array(c(3L, 2L, 4L)))
      autotest_vectorize(function(x) prim(x, axes = 1:2, drop = FALSE), x = rand_array(c(3L, 2L, 4L)))
    }
  })

  it("vectorizes prim_any and prim_all", {
    withr::local_seed(1L)
    for (prim in list(prim_any, prim_all)) {
      autotest_vectorize(
        function(x) prim(x, axes = 1L, drop = TRUE),
        x = rand_array(c(3L, 2L, 4L), dtype = "bool", sample = coin)
      )
    }
  })

  it("vectorizes prim_which_max and prim_which_min", {
    withr::local_seed(1L)
    for (prim in list(prim_which_max, prim_which_min)) {
      autotest_vectorize(function(x) prim(x, axis = 2L), x = rand_array(c(3L, 2L, 4L)))
    }
  })
})

describe("cumulative primitives", {
  it("vectorizes prim_cumsum, prim_cumprod, prim_cummax and prim_cummin", {
    withr::local_seed(1L)
    for (prim in list(prim_cumsum, prim_cumprod, prim_cummax, prim_cummin)) {
      autotest_vectorize(function(x) prim(x, axis = 2L), x = rand_array(c(3L, 2L, 4L)))
    }
  })
})

describe("prim_rev", {
  it("vectorizes", {
    autotest_vectorize(function(x) prim_rev(x, axes = 2L), x = rand_array(c(3L, 2L, 4L)))
  })
})

describe("prim_transpose", {
  it("vectorizes", {
    autotest_vectorize(function(x) prim_transpose(x, perm = c(2L, 1L)), x = rand_array(c(3L, 2L, 4L)))
  })
})

describe("prim_reshape", {
  it("vectorizes", {
    autotest_vectorize(function(x) prim_reshape(x, shape = c(4L, 2L)), x = rand_array(c(3L, 2L, 4L)))
  })
})

describe("prim_broadcast_in_axes", {
  it("vectorizes", {
    autotest_vectorize(
      function(x) prim_broadcast_in_axes(x, shape = c(2L, 5L), broadcast_axes = 1L),
      x = rand_array(c(3L, 2L))
    )
  })
})

describe("prim_concatenate", {
  it("vectorizes", {
    autotest_vectorize(
      function(x, y) prim_concatenate(x, y, axis = 1L),
      x = rand_array(c(3L, 2L, 2L)),
      y = rand_array(c(3L, 1L, 2L))
    )
  })
})

describe("prim_static_slice", {
  it("vectorizes", {
    autotest_vectorize(
      function(x) prim_static_slice(x, start_indices = c(1L, 2L), end_indices = c(2L, 4L), strides = c(1L, 2L)),
      x = rand_array(c(3L, 2L, 4L))
    )
  })
})

describe("prim_pad", {
  it("vectorizes", {
    autotest_vectorize(
      function(x) prim_pad(x, 0, c(1L, 0L), c(0L, 1L), c(1L, 0L)),
      x = rand_array(c(3L, 2L, 4L))
    )
  })
})

describe("prim_dynamic_slice", {
  it("vectorizes", {
    autotest_vectorize(
      function(x) prim_dynamic_slice(x, 2L, nv_scalar(1L, dtype = "i32"), slice_sizes = c(1L, 2L)),
      x = rand_array(c(3L, 2L, 4L))
    )
  })
})

describe("prim_dynamic_update_slice", {
  it("vectorizes", {
    autotest_vectorize(
      function(x, update) prim_dynamic_update_slice(x, update, 1L, 2L),
      x = rand_array(c(3L, 2L, 4L)),
      update = rand_array(c(3L, 1L, 2L))
    )
  })
})

describe("prim_sort", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, y) prim_sort(list(x, y), axis = 2L),
      x = rand_array(c(3L, 2L, 4L)),
      y = rand_array(c(3L, 2L, 4L))
    )
  })
})

describe("prim_top_k", {
  it("vectorizes", {
    withr::local_seed(1L)
    autotest_vectorize(function(x) prim_top_k(x, k = 2L, indices = TRUE), x = rand_array(c(3L, 2L, 4L)))
  })
})

describe("prim_dot_general", {
  it("vectorizes a matrix-vector product", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(lhs, rhs) {
        prim_dot_general(lhs, rhs, contracting_axes = list(2L, 1L), batching_axes = list(integer(), integer()))
      },
      lhs = rand_array(c(3L, 2L, 4L)),
      rhs = rand_array(c(3L, 4L))
    )
  })

  it("vectorizes a product with batching axes of its own", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(lhs, rhs) {
        prim_dot_general(lhs, rhs, contracting_axes = list(3L, 2L), batching_axes = list(1L, 1L))
      },
      lhs = rand_array(c(3L, 2L, 4L, 5L)),
      rhs = rand_array(c(3L, 2L, 5L, 6L))
    )
  })
})

describe("prim_gather", {
  it("vectorizes a gather of single elements", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, idx) {
        prim_gather(
          x,
          idx,
          slice_sizes = 1L,
          offset_axes = integer(),
          collapsed_slice_axes = 1L,
          x_batching_axes = integer(),
          start_indices_batching_axes = integer(),
          start_index_map = 1L,
          index_vector_axis = 2L
        )
      },
      x = rand_array(c(3L, 5L)),
      idx = rand_array(c(3L, 2L, 1L), dtype = "i32", sample = \(n) sample(1:5, n, replace = TRUE))
    )
  })

  it("vectorizes a gather of rows", {
    withr::local_seed(1L)
    autotest_vectorize(
      function(x, idx) {
        prim_gather(
          x,
          idx,
          slice_sizes = c(1L, 3L),
          offset_axes = 2L,
          collapsed_slice_axes = 1L,
          x_batching_axes = integer(),
          start_indices_batching_axes = integer(),
          start_index_map = 1L,
          index_vector_axis = 2L
        )
      },
      x = rand_array(c(3L, 4L, 3L)),
      idx = rand_array(c(3L, 2L, 1L), dtype = "i32", sample = \(n) sample(1:4, n, replace = TRUE))
    )
  })
})

describe("prim_chol", {
  it("vectorizes", {
    withr::local_seed(1L)
    spd <- vapply(
      1:3,
      function(i) {
        m <- matrix(stats::rnorm(9L), 3L)
        crossprod(m) + diag(3)
      },
      numeric(9L)
    )
    x <- nv_array(as.vector(t(spd)), shape = c(3L, 3L, 3L), dtype = "f32")
    autotest_vectorize(function(x) prim_chol(x, lower = TRUE), x = x, tolerance = 1e-4)
  })
})

describe("prim_triangular_solve", {
  it("vectorizes", {
    withr::local_seed(1L)
    lower <- vapply(
      1:3,
      function(i) {
        m <- matrix(stats::rnorm(9L), 3L)
        m[upper.tri(m)] <- 0
        m + 3 * diag(3)
      },
      numeric(9L)
    )
    a <- nv_array(as.vector(t(lower)), shape = c(3L, 3L, 3L), dtype = "f32")
    autotest_vectorize(
      function(a, b) {
        prim_triangular_solve(a, b, left_side = TRUE, lower = TRUE, unit_diagonal = FALSE, transpose_a = FALSE)
      },
      a = a,
      b = rand_array(c(3L, 3L, 2L)),
      tolerance = 1e-4
    )
  })
})

describe("prim_print", {
  it("vectorizes", {
    capture.output(autotest_vectorize(prim_print, x = rand_array(c(3L, 2L))))
  })
})
