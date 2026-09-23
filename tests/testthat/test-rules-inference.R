describe("assert_array_dtype()", {
  it("names the categories it wanted and the data type it got", {
    expect_snapshot(
      error = TRUE,
      assert_array_dtype(nv_aval("i32", 3L), "float", arg = "x")
    )
  })

  it("accepts an array in any one of several categories", {
    expect_silent(assert_array_dtype(nv_aval("ui8", 2L), "float", "uint"))
  })

  it("checks shape and rank when asked", {
    expect_snapshot(
      error = TRUE,
      assert_array_dtype(nv_aval("f32", c(2L, 3L)), shape = integer(), arg = "pred")
    )
    expect_snapshot(
      error = TRUE,
      assert_array_dtype(nv_aval("f32", c(2L, 3L)), naxes = 1L, arg = "initial_state")
    )
  })
})

describe("assert_arrays()", {
  it("names each operand by the argument it arrived in", {
    expect_error(
      assert_arrays(lhs = nv_aval("f32", integer()), rhs = 1L),
      "`rhs` must be an array",
      fixed = TRUE
    )
  })

  it("indexes the argument a primitive collects its operands in", {
    expect_error(
      assert_arrays(nv_aval("f32", integer()), 1L, .arg = "xs"),
      "`xs[[2]]` must be an array",
      fixed = TRUE
    )
  })

  it("names the caller's dots when the primitive takes them itself", {
    expect_error(
      assert_arrays(nv_aval("f32", integer()), 1L),
      "`..2` must be an array",
      fixed = TRUE
    )
  })
})

describe("test_permutation()", {
  it("rejects a repeated entry that a set comparison would accept", {
    # `setequal(c(1, 2, 2), 1:2)` is TRUE, which is why `setequal()` alone is
    # not enough: the length and the multiplicity both matter.
    expect_false(test_permutation(c(1L, 2L, 2L), 1:2))
    expect_false(test_permutation(1L, 1:2))
    expect_true(test_permutation(c(2L, 1L), 1:2))
  })
})

describe("the element-wise rules", {
  it("pass the operand type through unchanged", {
    x <- nv_aval("f32", c(2L, 3L))
    expect_equal(infer_generic_uni(x), list(x))
    expect_equal(infer_generic_biv(x, x), list(x))
  })

  it("refuse operands whose types disagree", {
    expect_snapshot(
      error = TRUE,
      infer_generic_biv(nv_aval("i32", 4L), nv_aval("i32", 6L))
    )
  })

  it("hold the bit shifts to integers, where booleans are not integers", {
    # StableHLO's `tensor of integer type` does not include `i1`, unlike the
    # bitwise `and` / `or` / `xor`, which take a `tensor of integer or boolean`.
    b <- nv_aval("bool", 2L)
    expect_equal(infer_integerish_biv(b, b), list(b))
    expect_error(infer_integer_biv(b, b), "must have an integer or unsigned integer data type")
  })

  it("give a boolean result for a predicate", {
    expect_equal(
      infer_is_finite(nv_aval("f32", c(2L, 3L))),
      list(nv_aval("bool", c(2L, 3L)))
    )
    expect_equal(
      infer_compare(nv_aval("i32", 4L), nv_aval("i32", 4L)),
      list(nv_aval("bool", 4L))
    )
  })
})

describe("infer_transpose()", {
  it("permutes the shape with 1-based axes", {
    expect_equal(
      infer_transpose(nv_aval("f32", c(2L, 3L, 4L)), c(3L, 1L, 2L)),
      list(nv_aval("f32", c(4L, 2L, 3L)))
    )
  })

  it("reports the permutation it expected in 1-based axes", {
    expect_snapshot(
      error = TRUE,
      infer_transpose(nv_aval("f32", c(2L, 2L)), 1L)
    )
  })
})

describe("infer_broadcast_in_axes()", {
  it("broadcasts a size-1 axis and keeps a matching one", {
    expect_equal(
      infer_broadcast_in_axes(nv_aval("f32", c(1L, 3L)), c(2L, 3L), c(1L, 2L)),
      list(nv_aval("f32", c(2L, 3L)))
    )
  })

  it("refuses an axis that is neither 1 nor the target size", {
    expect_snapshot(
      error = TRUE,
      infer_broadcast_in_axes(nv_aval("f32", c(2L, 3L)), c(4L, 3L), c(1L, 2L))
    )
  })
})

describe("infer_static_slice()", {
  it("treats start and end as 1-based and inclusive", {
    expect_equal(
      infer_static_slice(nv_aval("i32", 10L), 2L, 5L, 1L),
      list(nv_aval("i32", 4L))
    )
    expect_equal(
      infer_static_slice(nv_aval("i32", 10L), 1L, 10L, 2L),
      list(nv_aval("i32", 5L))
    )
  })

  it("refuses a stride of zero rather than computing an infinite shape", {
    # `ceiling(x / 0)` is `Inf`, which MLIR would only reject much later, after
    # an R coercion warning.
    expect_snapshot(
      error = TRUE,
      infer_static_slice(nv_aval("i32", 10L), 1L, 5L, 0L)
    )
  })

  it("refuses an end index past the end of the array", {
    expect_snapshot(
      error = TRUE,
      infer_static_slice(nv_aval("i32", 10L), 1L, 11L, 1L)
    )
  })
})

describe("infer_concatenate()", {
  it("sums the sizes along the concatenation axis", {
    x <- nv_aval("f32", c(2L, 3L))
    expect_equal(
      infer_concatenate(x, x, axis = 1L),
      list(nv_aval("f32", c(4L, 3L)))
    )
  })

  it("refuses inputs that disagree on any other axis", {
    expect_snapshot(
      error = TRUE,
      infer_concatenate(nv_aval("f32", c(2L, 3L)), nv_aval("f32", c(2L, 4L)), axis = 1L)
    )
  })

  it("refuses inputs with a different number of axes", {
    # Dropping `axis` from a shape that does not reach it leaves the shape
    # alone, so this passed the shape check and (C6) then indexed past the end
    # of the shorter one.
    expect_snapshot(
      error = TRUE,
      infer_concatenate(nv_aval("f32", c(2L, 3L, 4L)), nv_aval("f32", c(2L, 3L)), axis = 3L)
    )
  })
})

describe("infer_sort()", {
  it("mirrors each input, key and payloads alike", {
    key <- nv_aval("f32", c(2L, 3L))
    payload <- nv_aval("i32", c(2L, 3L))
    expect_equal(
      infer_sort(key, payload, axis = 1L, decreasing = FALSE, stable = FALSE),
      list(key, payload)
    )
  })

  it("names `xs`, the argument the primitive collects the arrays in", {
    expect_error(
      infer_sort(nv_aval("f32", 3L), 1L, axis = 1L, decreasing = FALSE, stable = FALSE),
      "`xs[[2]]` must be an array",
      fixed = TRUE
    )
    expect_error(
      infer_sort(axis = 1L, decreasing = FALSE, stable = FALSE),
      "`xs` must be a non-empty list",
      fixed = TRUE
    )
  })
})

describe("infer_dot_general()", {
  it("puts the batch axes first, then the free axes of lhs and rhs", {
    lhs <- nv_aval("f32", c(5L, 2L, 3L))
    rhs <- nv_aval("f32", c(5L, 3L, 7L))
    expect_equal(
      infer_dot_general(
        lhs,
        rhs,
        contracting_axes = list(3L, 2L),
        batching_axes = list(1L, 1L),
        precision = "highest"
      ),
      list(nv_aval("f32", c(5L, 2L, 7L)))
    )
  })

  it("refuses contracted axes whose sizes differ", {
    expect_snapshot(
      error = TRUE,
      infer_dot_general(
        nv_aval("f32", c(2L, 3L)),
        nv_aval("f32", c(4L, 5L)),
        contracting_axes = list(2L, 1L),
        batching_axes = list(integer(), integer()),
        precision = "highest"
      )
    )
  })
})

describe("infer_pad()", {
  it("accounts for edge and interior padding", {
    expect_equal(
      infer_pad(nv_aval("f32", 3L), nv_aval("f32", integer()), 2L, 1L, 0L),
      list(nv_aval("f32", 6L))
    )
    # Interior padding goes between elements, so `n - 1` gaps.
    expect_equal(
      infer_pad(nv_aval("f32", 3L), nv_aval("f32", integer()), 0L, 0L, 1L),
      list(nv_aval("f32", 5L))
    )
  })

  it("refuses negative padding that would empty an axis", {
    expect_snapshot(
      error = TRUE,
      infer_pad(nv_aval("f32", 3L), nv_aval("f32", integer()), -3L, -3L, 0L)
    )
  })
})

describe("reduced_shape()", {
  it("leaves the shape alone when no axes are reduced", {
    # `x[-integer(0)]` is `integer(0)`, not `x`, so an empty `axes` has to be a
    # special case: it once collapsed the result to a scalar while the emitted
    # program still produced the full shape.
    x <- nv_aval("f32", c(2L, 3L))
    expect_equal(reduced_shape(x, integer(0), drop = TRUE), c(2L, 3L))
    expect_equal(reduced_shape(x, integer(0), drop = FALSE), c(2L, 3L))
    expect_equal(reduced_shape(x, 1L, drop = TRUE), 3L)
    expect_equal(reduced_shape(x, 1L, drop = FALSE), c(1L, 3L))
  })

  it("reduces nothing, through the primitives, when axes is empty", {
    x <- nv_array(matrix(1:6, 2, 3), dtype = "f32")
    expect_equal(shape(prim_sum(x, axes = integer(0), drop = TRUE)), c(2L, 3L))
    expect_equal(
      as_array(prim_reduce(
        x,
        nv_scalar(0, "f32"),
        axes = integer(0),
        drop = TRUE,
        reducer = prim_add
      )),
      as_array(x)
    )
  })
})

describe("infer_convolution()", {
  it("refuses a padding that is not an (n_spatial, 2) matrix", {
    # `matrix(padding, nrow = n_spatial, ncol = 2L)` recycles or truncates to
    # that shape, so this has to be checked before the matrix is built.
    conv <- function(padding) {
      infer_convolution(
        nv_aval("f32", c(1L, 1L, 5L)),
        nv_aval("f32", c(1L, 1L, 3L)),
        1L,
        2L,
        3L,
        2L,
        1L,
        3L,
        1L,
        2L,
        3L,
        window_strides = 1L,
        padding = padding,
        x_dilation = 1L,
        kernel_dilation = 1L,
        feature_group_count = 1L,
        batch_group_count = 1L,
        precision = "highest"
      )
    }
    expect_equal(conv(matrix(0L, 1L, 2L)), list(nv_aval("f32", c(1L, 1L, 3L))))
    expect_error(conv(matrix(0L, 3L, 2L)), "must be a matrix of shape")
    expect_error(conv(c(1L, 2L)), "Got a vector of length 2")
  })
})

describe("infer_top_k()", {
  it("returns the values at the input's type and the indices at the default integer", {
    out <- infer_top_k(nv_aval("f32", c(2L, 8L)), k = 3L, indices = TRUE)
    expect_named(out, c("values", "indices"))
    expect_equal(out$values, nv_aval("f32", c(2L, 8L - 5L)))
    expect_equal(dtype(out$indices), default_int())
  })

  it("returns the values alone when the indices are not asked for", {
    out <- infer_top_k(nv_aval("f32", c(2L, 8L)), k = 3L, indices = FALSE)
    expect_named(out, "values")
    expect_equal(out$values, nv_aval("f32", c(2L, 3L)))
  })

  it("refuses a k larger than the last axis", {
    expect_snapshot(error = TRUE, infer_top_k(nv_aval("f32", c(2L, 3L)), k = 4L, indices = TRUE))
  })
})

describe("the inference rules as the primitives reach them", {
  it("reports an axis the primitive itself does not catch in 1-based terms", {
    # A too-short `permutation` passes `resolve_axes()` -- every entry is a
    # valid, non-duplicated axis -- and is only rejected by inference.
    expect_snapshot(
      error = TRUE,
      jit(prim_transpose, static = "perm")(
        nv_array(1:4, shape = c(2, 2)),
        perm = 1L
      )
    )
  })

  it("accepts an empty static slice, where start is one past end", {
    # 1-based and inclusive, so `start == end + 1` is StableHLO's
    # `start == limit`: an empty slice, which the spec allows.
    expect_equal(
      shape(jit(prim_static_slice, static = 2:4)(
        nv_array(1:10),
        start_indices = 6L,
        end_indices = 5L,
        strides = 1L
      )),
      0L
    )
  })

  it("names anvl's argument, not StableHLO's operand", {
    err <- tryCatch(jit(prim_ceiling)(nv_array(1:4)), error = identity)
    expect_match(conditionMessage(err), "`x`", fixed = TRUE)
    expect_false(grepl("operand", conditionMessage(err), fixed = TRUE))
  })

  it("speaks of arrays and axes, never tensors and dimensions", {
    err <- tryCatch(jit(prim_add)(nv_array(1:4), nv_array(1:6)), error = identity)
    msg <- conditionMessage(err)
    expect_false(grepl("tensor", msg, ignore.case = TRUE))
    expect_false(grepl("dimension", msg, ignore.case = TRUE))
  })
})

describe("the static parameters a rule is handed", {
  it("refuses a missing value instead of letting it reach an `if ()`", {
    # An `NA` used to reach the first comparison and come back out as R's own
    # "missing value where TRUE/FALSE needed", under the primitive's name.
    x <- nv_aval("f32", c(2L, 3L))
    expect_error(
      infer_pad(x, nv_aval("f32", integer()), c(NA_integer_, 0L), c(0L, 0L), c(0L, 0L)),
      "`edge_padding_low` must not contain missing values",
      fixed = TRUE
    )
    expect_error(
      jit(prim_pad, static = 3:5)(
        nv_array(1:4),
        nv_scalar(0L),
        NA_integer_,
        0L,
        0L
      ),
      "`edge_padding_low` must not contain missing values",
      fixed = TRUE
    )
  })

  it("refuses a value that is not a whole number", {
    x <- nv_aval("f32", 6L)
    expect_error(infer_reshape(x, "a"), "`shape` must be a whole number vector", fixed = TRUE)
    expect_error(infer_cum(nv_aval("f32", 3L), NULL), "`axis` must have 1 entry", fixed = TRUE)
    expect_error(infer_iota(1L, "f32", c(2L, 3L), "a"), "`start` must be a whole number", fixed = TRUE)
  })

  it("still reads `c()` as the empty set of axes", {
    # `c()` is `NULL`, and it is how a caller spells "no axes"; the old bare
    # `as.integer()` accepted it, so the guards must not refuse it.
    expect_silent(infer_reverse(nv_aval("f32", c(2L, 3L)), c()))
    expect_silent(infer_reduce_simple(nv_aval("f32", c(2L, 3L)), c(), TRUE))
    # A parameter that must be a single axis still refuses it.
    expect_error(infer_top_k(nv_aval("f32", 3L), NULL), "`k` must have 1 entry", fixed = TRUE)
  })

  it("refuses a negative size rather than letting Shape() complain", {
    expect_error(
      infer_reshape(nv_aval("f32", 6L), c(-6L, -1L)),
      "`shape` must not be negative",
      fixed = TRUE
    )
  })

  it("refuses a data type it cannot resolve, naming the argument", {
    expect_error(infer_convert(nv_aval("f32", integer()), "nope"), "`dtype` must name a data type", fixed = TRUE)
  })

  it("refuses a flag that is not one", {
    expect_error(infer_round(nv_aval("f32", 3L), "bogus"), "`method` must be", fixed = TRUE)
    expect_error(
      infer_cholesky(nv_aval("f32", c(2L, 2L)), NULL),
      "`lower` must be",
      fixed = TRUE
    )
  })
})

describe("infer_cond()", {
  it("reports a branch mismatch itself, rather than leaving it to the compiler", {
    expect_error(
      jit(function(p, a) {
        prim_if(p, function() a, function() nv_scalar(1L, dtype = "i32"))
      })(nv_scalar(TRUE), nv_scalar(1.0)),
      "`true` and `false` must return the same type",
      fixed = TRUE
    )
  })
})

describe("the reduce rules", {
  it("own their axis contract, as infer_reduce does", {
    x <- nv_aval("f32", c(2L, 3L))
    expect_error(infer_reduce_simple(x, 5L, TRUE), "must contain axes between 1 and 2", fixed = TRUE)
    expect_error(infer_reduce_simple(x, c(1L, 1L), TRUE), "must contain unique axes", fixed = TRUE)
    expect_error(
      infer_reduce_boolean(nv_aval("bool", c(2L, 3L)), 5L, TRUE),
      "must contain axes between 1 and 2",
      fixed = TRUE
    )
  })
})

describe("the rules that take an array but are handed an R value", {
  it("say so, instead of describing it as a scalar or a rank-0 matrix", {
    expect_error(infer_qr(42), "`x` must be an array", fixed = TRUE)
    expect_error(infer_cum(42, 1L), "`x` must be an array", fixed = TRUE)
    expect_error(infer_arg_extreme(42, 1L, TRUE), "`x` must be an array", fixed = TRUE)
  })
})

describe("infer_triangular_solve()", {
  it("refuses a flag it would otherwise branch on", {
    # `side <- if (left_side) ...` turned an `NA` into R's own
    # "missing value where TRUE/FALSE needed", under `prim_triangular_solve()`.
    a <- nv_aval("f32", c(2L, 2L))
    b <- nv_aval("f32", c(2L, 1L))
    for (flag in c("left_side", "lower", "unit_diagonal", "transpose_a")) {
      args <- list(a, b, TRUE, TRUE, FALSE, FALSE)
      names(args) <- c("a", "b", "left_side", "lower", "unit_diagonal", "transpose_a")
      args[[flag]] <- NA
      expect_error(
        do.call(infer_triangular_solve, args),
        paste0("`", flag, "` must be"),
        fixed = TRUE,
        info = flag
      )
    }
  })
})

describe("infer_convolution()", {
  it("requires each layout to account for every axis", {
    expect_error(
      jit(prim_convolution, static = 3:18)(
        nv_array(as.double(1:25), shape = c(1, 1, 5, 5)),
        nv_array(as.double(1:9), shape = c(1, 1, 3, 3)),
        integer(),
        2L,
        c(3L, 4L),
        2L,
        1L,
        c(3L, 4L),
        1L,
        2L,
        c(3L, 4L),
        c(1L, 1L),
        matrix(0L, 2L, 2L),
        c(1L, 1L),
        c(1L, 1L),
        1L,
        1L,
        "highest"
      ),
      "must each be named exactly once",
      fixed = TRUE
    )
  })

  it("checks `padding`, which keeps its shape and so skips the usual guard", {
    conv <- function(padding) {
      jit(prim_convolution, static = 3:18)(
        nv_array(as.double(1:25), shape = c(1, 1, 5, 5)),
        nv_array(as.double(1:9), shape = c(1, 1, 3, 3)),
        1L,
        2L,
        c(3L, 4L),
        2L,
        1L,
        c(3L, 4L),
        1L,
        2L,
        c(3L, 4L),
        c(1L, 1L),
        padding,
        c(1L, 1L),
        c(1L, 1L),
        1L,
        1L,
        "highest"
      )
    }
    expect_error(conv(matrix(NA_integer_, 2L, 2L)), "`padding` must not contain missing values", fixed = TRUE)
    expect_error(conv(matrix(0.5, 2L, 2L)), "`padding` must contain whole numbers", fixed = TRUE)
  })
})

# Every message a rule refuses a param with names the param and reports the
# value it was handed. Each one is snapshotted through the primitive itself, so
# that the snapshot holds the message a caller actually sees. A param a wrapper
# resolves before the rule sees it (`resolve_axes()`, `assert_shapevec()`) is
# answered there instead, in the same terms.

test_that("prim_reshape", {
  expect_snapshot(error = TRUE, prim_reshape(nv_array(1:4), shape = "a"))
  expect_snapshot(error = TRUE, prim_reshape(nv_array(1:4), shape = c(3L, 3L)))
})

test_that("prim_rev", {
  expect_snapshot(error = TRUE, prim_rev(nv_array(1:4), axes = list(1L)))
})

test_that("prim_cumsum", {
  expect_snapshot(error = TRUE, prim_cumsum(nv_array(1:4), axis = c(1L, 1L)))
})

test_that("prim_sum", {
  expect_snapshot(error = TRUE, prim_sum(nv_array(1:4), axes = 1L, drop = "yes"))
  expect_snapshot(error = TRUE, prim_sum(nv_array(1:4), axes = 1L, drop = NA))
  expect_snapshot(error = TRUE, prim_sum(nv_array(1:4), axes = "a"))
})

test_that("prim_convert", {
  expect_snapshot(error = TRUE, prim_convert(nv_array(1:4), dtype = "nope"))
  expect_snapshot(error = TRUE, prim_convert(nv_array(1:4), dtype = 42))
})

test_that("prim_round", {
  expect_snapshot(error = TRUE, prim_round(nv_array(c(1.5, 2.5)), method = "bogus"))
})

test_that("prim_rng_bit_generator", {
  state <- nv_array(c(1, 2), dtype = "ui64")
  expect_snapshot(error = TRUE, prim_rng_bit_generator(state, "MERSENNE", "f32", 3L))
})

test_that("prim_fill", {
  expect_snapshot(error = TRUE, prim_fill(c(1, 2), 3L, "f32"))
  expect_snapshot(error = TRUE, prim_fill(1, 3L, "nope"))
})

test_that("prim_iota", {
  expect_snapshot(error = TRUE, prim_iota(axis = 1L, dtype = "f32", shape = "a"))
  expect_snapshot(error = TRUE, prim_iota(axis = 1L, dtype = "bool", shape = 3L))
})

test_that("prim_broadcast_in_axes", {
  x <- nv_array(as.double(1:12), shape = c(4, 3))
  expect_snapshot(
    error = TRUE,
    prim_broadcast_in_axes(x, shape = c(4L, 3L), broadcast_axes = 1L)
  )
})

test_that("prim_static_slice", {
  x <- nv_array(as.double(1:12), shape = c(4, 3))
  expect_snapshot(error = TRUE, prim_static_slice(x, 1L, c(2L, 2L), 1L))
  expect_snapshot(error = TRUE, prim_static_slice(x, c(1L, 1L), c(2L, 2L), c(0L, 1L)))
})

test_that("prim_pad", {
  x <- nv_array(as.double(1:12), shape = c(4, 3))
  expect_snapshot(error = TRUE, prim_pad(x, nv_scalar(0), 0L, c(0L, 0L), c(0L, 0L)))
  expect_snapshot(
    error = TRUE,
    prim_pad(nv_array(as.double(1:3)), nv_scalar(0), -3L, -3L, 0L)
  )
})

test_that("prim_dynamic_slice", {
  x <- nv_array(as.double(1:12), shape = c(4, 3))
  expect_snapshot(
    error = TRUE,
    prim_dynamic_slice(x, nv_scalar(1L), nv_scalar(1L), slice_sizes = 2L)
  )
})

test_that("prim_top_k", {
  expect_snapshot(error = TRUE, prim_top_k(nv_array(1:4), k = 1L, indices = "yes"))
  expect_snapshot(error = TRUE, prim_top_k(nv_array(1:4), k = c(1L, 2L)))
})

test_that("prim_chol", {
  m <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
  expect_snapshot(error = TRUE, prim_chol(m, lower = "x"))
})

test_that("prim_sort", {
  # The rule cannot answer for `xs`: the wrapper's next line resolves `axis`
  # against `xs[[1L]]`. It answers in the rule's terms instead.
  expect_snapshot(error = TRUE, prim_sort(list(), axis = 1L))
  expect_snapshot(error = TRUE, prim_sort(list(nv_array(1:4)), axis = 1L, decreasing = "yes"))
})

test_that("prim_dot_general", {
  lhs <- nv_array(as.double(1:6), shape = c(2, 3))
  rhs <- nv_array(as.double(1:12), shape = c(3, 4))
  expect_snapshot(
    error = TRUE,
    prim_dot_general(lhs, rhs, contracting_axes = 1L, batching_axes = list(integer(), integer()))
  )
  expect_snapshot(
    error = TRUE,
    prim_dot_general(
      lhs,
      rhs,
      contracting_axes = list(2L, 1L),
      batching_axes = list(1L, integer())
    )
  )
  expect_snapshot(
    error = TRUE,
    prim_dot_general(
      lhs,
      rhs,
      contracting_axes = list(c(1L, 2L), 1L),
      batching_axes = list(integer(), integer())
    )
  )
  expect_snapshot(
    error = TRUE,
    prim_dot_general(
      lhs,
      rhs,
      contracting_axes = list(2L, 1L),
      batching_axes = list(integer(), integer()),
      precision = "bogus"
    )
  )
})

test_that("prim_gather", {
  gather <- function(
    slice_sizes = c(1L, 3L),
    collapsed_slice_axes = 1L,
    x_batching_axes = integer(),
    start_index_map = 1L
  ) {
    prim_gather(
      nv_array(as.double(1:12), shape = c(4, 3)),
      nv_array(matrix(c(1L, 2L), nrow = 2, ncol = 1)),
      slice_sizes = slice_sizes,
      offset_axes = 2L,
      collapsed_slice_axes = collapsed_slice_axes,
      x_batching_axes = x_batching_axes,
      start_indices_batching_axes = integer(),
      start_index_map = start_index_map,
      index_vector_axis = 2L,
      indices_are_sorted = FALSE,
      unique_indices = FALSE
    )
  }
  expect_equal(shape(gather()), c(2L, 3L))
  expect_snapshot(error = TRUE, gather(slice_sizes = c(1L, 3L, 1L)))
  expect_snapshot(error = TRUE, gather(start_index_map = c(1L, 2L)))
  expect_snapshot(
    error = TRUE,
    gather(collapsed_slice_axes = integer(), x_batching_axes = 1L)
  )
})

test_that("prim_scatter", {
  scatter <- function(inserted_window_axes = 1L, x_batching_axes = integer(), scatter_axes_to_x_axes = 1L) {
    prim_scatter(
      nv_array(as.double(1:12), shape = c(4, 3)),
      nv_array(matrix(c(1L, 2L), nrow = 2, ncol = 1)),
      nv_array(as.double(1:6), shape = c(2, 3)),
      update_window_axes = 2L,
      inserted_window_axes = inserted_window_axes,
      x_batching_axes = x_batching_axes,
      scatter_indices_batching_axes = integer(),
      scatter_axes_to_x_axes = scatter_axes_to_x_axes,
      index_vector_axis = 2L,
      indices_are_sorted = FALSE,
      unique_indices = FALSE,
      update_fn = function(a, b) b
    )
  }
  expect_equal(shape(scatter()), c(4L, 3L))
  expect_snapshot(error = TRUE, scatter(scatter_axes_to_x_axes = c(1L, 2L)))
  expect_snapshot(
    error = TRUE,
    scatter(inserted_window_axes = integer(), x_batching_axes = 1L)
  )
})

test_that("prim_convolution", {
  conv <- function(window_strides = 1L, x_spatial_axes = 3L, x_batch_axis = 1L, precision = "highest") {
    prim_convolution(
      nv_array(as.double(1:5), shape = c(1, 1, 5)),
      nv_array(as.double(1:3), shape = c(1, 1, 3)),
      x_batch_axis = x_batch_axis,
      x_feature_axis = 2L,
      x_spatial_axes = x_spatial_axes,
      kernel_input_feature_axis = 2L,
      kernel_output_feature_axis = 1L,
      kernel_spatial_axes = 3L,
      output_batch_axis = 1L,
      output_feature_axis = 2L,
      output_spatial_axes = 3L,
      window_strides = window_strides,
      padding = matrix(0L, 1L, 2L),
      x_dilation = 1L,
      kernel_dilation = 1L,
      feature_group_count = 1L,
      batch_group_count = 1L,
      precision = precision
    )
  }
  expect_equal(shape(conv()), c(1L, 1L, 3L))
  expect_snapshot(error = TRUE, conv(window_strides = c(1L, 1L)))
  expect_snapshot(error = TRUE, conv(x_spatial_axes = c(3L, 4L)))
  expect_snapshot(error = TRUE, conv(x_batch_axis = 2L))
  expect_snapshot(error = TRUE, conv(x_batch_axis = integer()))
  expect_snapshot(error = TRUE, conv(precision = "bogus"))
})

describe("value_repr()", {
  # `cat()` so that the snapshot holds the text a message shows, unescaped.
  show_repr <- function(x) cat(value_repr(x), "\n", sep = "")

  it("spells a value the way format_param() does", {
    expect_snapshot({
      show_repr(NULL)
      show_repr(3.5)
      show_repr(TRUE)
      show_repr("afz")
      show_repr(c(1L, 3L))
      show_repr(c("a", "b"))
      show_repr(integer())
      show_repr(character())
    })
  })

  it("cuts a long vector or string short", {
    expect_snapshot({
      show_repr(1:8)
      show_repr(1:1000)
      show_repr(rep("afz", 1000))
      show_repr(strrep("a", 5000))
    })
  })

  it("reports an array by its array type, not the class it arrives in", {
    # Under `jit()` an operand is a `GraphBox`, which is the tracer's business
    # and not something the caller wrote.
    expect_equal(value_repr(nv_array(1:4, dtype = "i32")), "i32[4]")
    expect_error(
      prim_sort(nv_array(1:4, dtype = "i32"), axis = 1L),
      "Got i32[4]",
      fixed = TRUE
    )
  })

  it("copes with values format_param() never sees", {
    expect_snapshot({
      show_repr(NA)
      show_repr(NA_character_)
      show_repr(c(1L, NA))
      show_repr(NaN)
      show_repr(Inf)
      show_repr(matrix(1:4, 2))
      show_repr(matrix(1:1000, 10))
      show_repr(matrix(5L, 1, 1))
      show_repr(matrix(integer(), 0, 3))
      show_repr(array(1:8, c(2, 2, 2)))
      show_repr(array(1:3))
      show_repr(list())
      show_repr(as.list(1:1000))
      show_repr(factor(letters))
      show_repr(mean)
      show_repr(globalenv())
    })
  })

  it("is what a rule reports", {
    expect_error(assert_int_param("a", "axis"), 'Got "a"', fixed = TRUE)
    # `{.val {mean}}` errors inside the message it is meant to report.
    expect_error(assert_flag_param(mean, "drop"), "Got <function>", fixed = TRUE)
  })
})

describe("shape_repr()", {
  it("prints a real shape whole and a typed-in one cut short", {
    expect_equal(shape_repr(c(2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)), "(2x3x4x5x6x7x8x9)")
    expect_equal(shape_repr(1:1000), "(1x2x3x4x5x6x7x8x...) with 1000 axes")
  })
})

# A caller's value can be arbitrarily large -- a whole data vector passed where
# a scalar belongs. The message reports it cut short, and is quick to build.
test_that("messages stay short for oversized params", {
  expect_snapshot(error = TRUE, prim_fill(1:1000, 3L, "f32"))
  expect_snapshot(error = TRUE, prim_fill(1, 3L, strrep("a", 5000)))
  expect_snapshot(error = TRUE, prim_round(nv_array(c(1.5, 2.5)), method = rep("afz", 1000)))
  expect_snapshot(error = TRUE, prim_static_slice(nv_array(1:4), 1:1000, 1:1000, 1:1000))
  expect_snapshot(error = TRUE, prim_reshape(nv_array(1:4), shape = 1:1000))
  elapsed <- system.time(try(prim_fill(seq_len(1e6), 3L, "f32"), silent = TRUE))[["elapsed"]]
  expect_lt(elapsed, 1)
})

test_that("a shape with too many elements is refused before it reaches XLA", {
  expect_snapshot(error = TRUE, prim_fill(1, 1:1000, "f32"))
})

test_that("a whole-number param outside the integer range is not reported as NA", {
  expect_snapshot(error = TRUE, prim_top_k(nv_array(1:4), k = Inf))
})

test_that("broadcasting to a size-1 axis names 1 once", {
  x <- nv_array(as.double(1:12), shape = c(4, 3))
  expect_snapshot(error = TRUE, prim_broadcast_in_axes(x, shape = c(1L, 3L), broadcast_axes = 1:2))
})

describe("a result shape a rule computes from the caller's parameters", {
  # `prim_convolution()` is the path a caller takes, and the one that used to
  # reach XLA; the rule alone would not exercise the wrapper's promotion.
  conv <- function(...) {
    args <- list(
      x = nv_array(array(as.double(1:16), c(1L, 1L, 4L, 4L)), dtype = "f32"),
      kernel = nv_array(array(as.double(1:4), c(1L, 1L, 2L, 2L)), dtype = "f32"),
      x_batch_axis = 1L,
      x_feature_axis = 2L,
      x_spatial_axes = c(3L, 4L),
      kernel_input_feature_axis = 2L,
      kernel_output_feature_axis = 1L,
      kernel_spatial_axes = c(3L, 4L),
      output_batch_axis = 1L,
      output_feature_axis = 2L,
      output_spatial_axes = c(3L, 4L),
      window_strides = c(1L, 1L),
      padding = matrix(0L, 2L, 2L),
      x_dilation = c(1L, 1L),
      kernel_dilation = c(1L, 1L)
    )
    do.call(prim_convolution, utils::modifyList(args, list(...)))
  }

  it("refuses a negative padding that empties a spatial axis past zero", {
    # XLA's own shape inference `CHECK`-fails on the negative window bound this
    # produces, which aborts the R process rather than raising an error.
    expect_snapshot(error = TRUE, conv(padding = matrix(-100L, 2L, 2L)))
  })

  it("refuses it through the user-facing convolution too", {
    expect_error(
      nv_conv2d(
        nv_array(array(as.double(1:16), c(1L, 1L, 4L, 4L)), dtype = "f32"),
        nv_array(array(as.double(1:4), c(1L, 1L, 2L, 2L)), dtype = "f32"),
        padding = -100L
      ),
      "must not remove more than spatial axis",
      fixed = TRUE
    )
  })

  it("refuses a zero-sized kernel spatial axis", {
    # A zero-wide window reads as "not wider than the input" below, so the rule
    # would infer a non-empty result from it; StableHLO refuses it outright,
    # reporting the axis 0-based.
    expect_snapshot(
      error = TRUE,
      conv(kernel = nv_array(array(numeric(), c(1L, 1L, 0L, 0L)), dtype = "f32"))
    )
  })

  it("refuses an overflowing dilation or padding rather than reaching an `if ()` with an `NA`", {
    # Each entry is inside the integer range, so `assert_int_param()` accepts
    # it; the window arithmetic is what overflows.
    expect_snapshot(error = TRUE, conv(x_dilation = c(2000000000L, 1L)))
    expect_snapshot(error = TRUE, conv(padding = matrix(2000000000L, 2L, 2L)))
  })

  it("still accepts a large but legal dilation", {
    expect_equal(shape(conv(x_dilation = c(3L, 1L))), c(1L, 1L, 9L, 3L))
  })

  it("reports an out-of-range `end_indices` rather than overflowing on it", {
    # `end + 1L` at `.Machine$integer.max` overflows to `NA`, which used to
    # reach the `start > end + 1L` comparison, so the shape check runs first.
    expect_snapshot(
      error = TRUE,
      prim_static_slice(nv_array(1:4), 1L, .Machine$integer.max, 1L)
    )
  })

  it("still names `start_indices` when that is the mistake", {
    expect_error(
      prim_static_slice(nv_array(1:4), 3L, 1L, 1L),
      "`start_indices` must not exceed `end_indices`",
      fixed = TRUE
    )
  })
})

describe("a whole-number parameter the primitive fixes at one entry", {
  it("is spoken of in the singular by every branch that refuses it", {
    expect_snapshot(error = TRUE, prim_top_k(nv_array(1:4), NA))
    expect_snapshot(error = TRUE, prim_top_k(nv_array(1:4), 2.5))
    expect_snapshot(error = TRUE, prim_top_k(nv_array(1:4), 3e9))
  })

  it("leaves a vector parameter in the plural", {
    expect_error(
      prim_pad(nv_array(1:4), nv_scalar(1L), 0.5, 0L, 0L),
      "`edge_padding_low` must contain whole numbers",
      fixed = TRUE
    )
  })
})

describe("the sub-graph arguments a primitive traces", {
  it("refuses a `prim_if()` branch that is not a function", {
    # Without this the branch reaches `do.call()` inside the tracer, which
    # reports `'what' must be a function or character string`. `prim_while()`
    # already checked `cond` and `body` this way.
    expect_snapshot(
      error = TRUE,
      prim_if(nv_scalar(TRUE), 1L, function() nv_scalar(1L))
    )
    expect_snapshot(
      error = TRUE,
      prim_if(nv_scalar(TRUE), function() nv_scalar(1L), "x")
    )
  })
})
