test_that("quickr pipeline matches PJRT: core math + comparisons + reductions", {
  skip_if_no_quickr_or_pjrt()

  core_ops <- function(x_f64, y_f64, x_i32, y_i32, p, q) {
    a <- x_f64 + y_f64
    b <- x_f64 - y_f64
    c <- x_f64 * y_f64
    d <- x_f64 / (y_f64 + 1L)
    e <- nv_negate(x_f64)

    ax <- nv_abs(x_f64)
    sq <- nv_sqrt(ax + 0.1)
    lg <- nv_log(ax + 1.1)
    ex <- nv_exp(x_f64)
    em1 <- nv_expm1(x_f64)
    l1p <- nv_log1p(ax * 0.1)
    lo <- nv_logistic(x_f64)
    si <- nv_sin(x_f64)
    co <- nv_cos(x_f64)
    ta <- nv_tan(x_f64 * 0.1)
    th <- nv_tanh(x_f64)
    fl <- nv_floor(x_f64)
    ce <- nv_ceiling(x_f64)

    mx <- nv_max(x_f64, y_f64)
    mn <- nv_min(x_f64, y_f64)
    pw <- nv_pow(ax + 0.1, nv_abs(y_f64) + 0.1)

    eq <- nv_eq(x_i32, y_i32)
    ne <- nv_ne(x_i32, y_i32)
    gt <- nv_gt(x_i32, y_i32)
    ge <- nv_ge(x_i32, y_i32)
    lt <- nv_lt(x_i32, y_i32)
    le <- nv_le(x_i32, y_i32)

    aand <- nv_and(p, q)
    oor <- nv_or(p, q)
    xxor <- nv_xor(p, q)
    nn <- nv_not(p)

    sel <- nv_ifelse(eq, x_f64, y_f64)

    sm <- sum(x_f64)
    rs1 <- nv_reduce_sum(x_f64, dims = 1L, drop = TRUE)
    rs2 <- nv_reduce_sum(x_f64, dims = 2L, drop = FALSE)
    rp <- nv_reduce_prod(ax + 1.0, dims = 2L, drop = TRUE)
    ra <- nv_reduce_any(eq, dims = 1L, drop = TRUE)
    rall <- nv_reduce_all(eq, dims = c(1L, 2L), drop = TRUE)

    rmax <- nv_reduce_max(x_f64, dims = 2L, drop = TRUE)
    rmin <- nv_reduce_min(x_f64, dims = 1L, drop = TRUE)

    list(
      add = a,
      sub = b,
      mul = c,
      div = d,
      neg = e,
      abs = ax,
      sqrt = sq,
      log = lg,
      exp = ex,
      expm1 = em1,
      log1p = l1p,
      logistic = lo,
      sine = si,
      cosine = co,
      tan = ta,
      tanh = th,
      floor = fl,
      ceil = ce,
      maximum = mx,
      minimum = mn,
      power = pw,
      eq = eq,
      ne = ne,
      gt = gt,
      ge = ge,
      lt = lt,
      le = le,
      and = aand,
      or = oor,
      xor = xxor,
      not = nn,
      select = sel,
      sum = sm,
      reduce_sum_1 = rs1,
      reduce_sum_2 = rs2,
      reduce_prod = rp,
      reduce_any = ra,
      reduce_all = rall,
      reduce_max = rmax,
      reduce_min = rmin
    )
  }

  templates <- list(
    x_f64 = nv_matrix(0, nrow = 2L, ncol = 3L, dtype = "f64"),
    y_f64 = nv_matrix(0, nrow = 2L, ncol = 3L, dtype = "f64"),
    x_i32 = nv_matrix(0L, nrow = 2L, ncol = 3L, dtype = "i32"),
    y_i32 = nv_matrix(0L, nrow = 2L, ncol = 3L, dtype = "i32"),
    p = nv_scalar(FALSE, dtype = "pred"),
    q = nv_scalar(FALSE, dtype = "pred")
  )

  run1 <- list(
    args = list(
      x_f64 = matrix(c(-0.2, -0.1, -0.3, -0.4, -0.5, -0.6), nrow = 2, byrow = TRUE),
      y_f64 = matrix(c(0.9, -0.8, 0.7, 0.6, -0.5, 0.4), nrow = 2, byrow = TRUE),
      x_i32 = matrix(c(1L, 2L, 3L, 4L, 5L, 6L), nrow = 2, byrow = TRUE),
      y_i32 = matrix(c(1L, 0L, 3L, 9L, 5L, 7L), nrow = 2, byrow = TRUE),
      p = TRUE,
      q = FALSE
    ),
    info = "run1"
  )

  run2 <- list(
    args = list(
      x_f64 = matrix(c(0.2, 0.1, 0.3, 0.4, 0.5, 0.6), nrow = 2, byrow = TRUE),
      y_f64 = matrix(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), nrow = 2, byrow = TRUE),
      x_i32 = matrix(c(0L, 1L, 0L, 1L, 0L, 1L), nrow = 2, byrow = TRUE),
      y_i32 = matrix(c(1L, 1L, 0L, 0L, 1L, 1L), nrow = 2, byrow = TRUE),
      p = FALSE,
      q = FALSE
    ),
    info = "run2"
  )

  expect_quickr_matches_pjrt_fn(core_ops, templates, list(run1, run2), tolerance = 1e-6)
})

test_that("quickr pipeline matches PJRT: i32 divide truncates toward zero", {
  skip_if_no_quickr_or_pjrt()

  divide_i32 <- function(x, y) {
    x / y
  }

  graph <- trace_fn(
    divide_i32,
    list(
      x = nv_matrix(0L, nrow = 2L, ncol = 2L, dtype = "i32"),
      y = nv_matrix(1L, nrow = 2L, ncol = 2L, dtype = "i32")
    )
  )

  lhs <- matrix(c(7L, -7L, 7L, -7L), nrow = 2L, byrow = TRUE)
  rhs <- matrix(c(2L, 2L, -2L, -2L), nrow = 2L, byrow = TRUE)

  out_pjrt <- quickr_eval_graph_pjrt(graph, lhs, rhs)
  expect_identical(out_pjrt, matrix(c(3L, -3L, -3L, 3L), nrow = 2L, byrow = TRUE))

  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(f_r(lhs, rhs), out_pjrt)
  expect_identical(as_array(f_quick(lhs, rhs)), out_pjrt)
})

test_that("quickr pipeline matches PJRT: reshape unboxes singleton arrays to scalars", {
  skip_if_no_quickr_or_pjrt()

  reshape_to_scalar <- function(x) {
    nv_reshape(x, shape = integer())
  }

  graph <- trace_fn(
    reshape_to_scalar,
    list(x = nv_fill(0.0, shape = c(1L, 1L), dtype = "f64"))
  )

  x <- matrix(2.5, nrow = 1L, ncol = 1L)
  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  expect_identical(out_pjrt, 2.5)

  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(f_r(x), out_pjrt)
  expect_identical(as_array(f_quick(x)), out_pjrt)
})

test_that("quickr pipeline matches PJRT: tanh stays finite for large inputs", {
  skip_if_no_quickr_or_pjrt()

  tanh_large <- function(x) {
    nv_tanh(x)
  }

  graph <- trace_fn(
    tanh_large,
    list(x = nv_scalar(0.0, dtype = "f64"))
  )

  x <- 1000

  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  expect_true(is.finite(out_pjrt))

  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(f_r(x), out_pjrt)
  expect_identical(as_array(f_quick(x)), out_pjrt)
})

test_that("quickr pipeline matches PJRT: select preserves array semantics in plain R", {
  skip_if_no_quickr_or_pjrt()

  select_cases <- function(pred_scalar, x, y, pred_empty, x_empty, y_empty) {
    list(
      scalar_pred = nv_ifelse(pred_scalar, x, y),
      empty = nv_ifelse(pred_empty, x_empty, y_empty)
    )
  }

  graph <- trace_fn(
    select_cases,
    list(
      pred_scalar = nv_scalar(FALSE, dtype = "pred"),
      x = nv_array(rep(0L, 3L), shape = 3L, dtype = "i32"),
      y = nv_array(rep(0L, 3L), shape = 3L, dtype = "i32"),
      pred_empty = nv_array(rep(FALSE, 0L), shape = 0L, dtype = "pred"),
      x_empty = nv_array(rep(0L, 0L), shape = 0L, dtype = "i32"),
      y_empty = nv_array(rep(0L, 0L), shape = 0L, dtype = "i32")
    )
  )

  out_pjrt <- quickr_eval_graph_pjrt(
    graph,
    TRUE,
    array(c(1L, 2L, 3L), dim = 3L),
    array(c(4L, 5L, 6L), dim = 3L),
    array(logical(), dim = 0L),
    array(integer(), dim = 0L),
    array(integer(), dim = 0L)
  )

  expect_identical(out_pjrt$scalar_pred, array(c(1L, 2L, 3L), dim = 3L))
  expect_identical(out_pjrt$empty, array(integer(), dim = 0L))

  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph, unwrap = TRUE)

  expect_identical(
    f_r(
      TRUE,
      array(c(1L, 2L, 3L), dim = 3L),
      array(c(4L, 5L, 6L), dim = 3L),
      array(logical(), dim = 0L),
      array(integer(), dim = 0L),
      array(integer(), dim = 0L)
    ),
    out_pjrt
  )
  expect_identical(
    f_quick(
      TRUE,
      array(c(1L, 2L, 3L), dim = 3L),
      array(c(4L, 5L, 6L), dim = 3L),
      array(logical(), dim = 0L),
      array(integer(), dim = 0L),
      array(integer(), dim = 0L)
    ),
    out_pjrt
  )
})

test_that("quickr pipeline matches PJRT: expm1 stays accurate near zero", {
  skip_if_no_quickr_or_pjrt()

  expm1_small <- function(x) {
    nv_expm1(x)
  }

  graph <- trace_fn(
    expm1_small,
    list(x = nv_scalar(0.0, dtype = "f64"))
  )

  x <- 1e-16
  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  expect_identical(out_pjrt, x)

  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(f_r(x), out_pjrt)
  expect_identical(as_array(f_quick(x)), out_pjrt)
})

test_that("quickr pipeline matches PJRT: expm1/log1p preserve empty f64 dtype", {
  skip_if_no_quickr_or_pjrt()

  x <- array(double(), dim = 0L)

  cases <- list(
    list(
      fn = function(x) {
        nv_expm1(x)
      },
      info = "expm1"
    ),
    list(
      fn = function(x) {
        nv_log1p(x)
      },
      info = "log1p"
    )
  )

  for (case in cases) {
    graph <- trace_fn(
      case$fn,
      list(x = nv_array(x, dtype = "f64", shape = 0L))
    )

    out_pjrt <- quickr_eval_graph_pjrt(graph, x)
    expect_identical(out_pjrt, x, info = case$info)

    f_r <- graph_to_quickr_r_function(graph)
    f_quick <- graph_to_quickr_function(graph)

    expect_identical(f_r(x), out_pjrt, info = case$info)
    expect_identical(as_array(f_quick(x)), out_pjrt, info = case$info)
  }
})

test_that("quickr pipeline matches PJRT: log1p stays accurate near zero", {
  skip_if_no_quickr_or_pjrt()

  log1p_small <- function(x) {
    nv_log1p(x)
  }

  graph <- trace_fn(
    log1p_small,
    list(x = nv_scalar(0.0, dtype = "f64"))
  )

  x <- 1e-16
  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  expect_identical(out_pjrt, x)

  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(f_r(x), out_pjrt)
  expect_identical(as_array(f_quick(x)), out_pjrt)
})

test_that("quickr pipeline matches PJRT: maximum/minimum preserve empty f64 dtype", {
  skip_if_no_quickr_or_pjrt()

  x <- array(double(), dim = 0L)

  cases <- list(
    list(
      fn = function(x, y) {
        nv_max(x, y)
      },
      info = "maximum"
    ),
    list(
      fn = function(x, y) {
        nv_min(x, y)
      },
      info = "minimum"
    )
  )

  for (case in cases) {
    graph <- trace_fn(
      case$fn,
      list(
        x = nv_array(x, dtype = "f64", shape = 0L),
        y = nv_array(x, dtype = "f64", shape = 0L)
      )
    )

    out_pjrt <- quickr_eval_graph_pjrt(graph, x, x)
    expect_identical(out_pjrt, x, info = case$info)

    f_r <- graph_to_quickr_r_function(graph)
    f_quick <- graph_to_quickr_function(graph)

    expect_identical(f_r(x, x), out_pjrt, info = case$info)
    expect_identical(as_array(f_quick(x, x)), out_pjrt, info = case$info)
  }
})

test_that("quickr pipeline matches PJRT: pad supports negative edge padding crop", {
  skip_if_no_quickr_or_pjrt()

  pad_crop <- function(x) {
    nv_pad(
      x,
      nv_scalar(0L, dtype = "i32"),
      edge_padding_low = c(-1L, 1L),
      edge_padding_high = c(0L, 0L),
      interior_padding = c(0L, 0L)
    )
  }

  graph <- trace_fn(
    pad_crop,
    list(x = nv_fill(0L, shape = c(2L, 2L), dtype = "i32"))
  )

  x <- matrix(c(1L, 2L, 3L, 4L), nrow = 2L, byrow = TRUE)

  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  expect_identical(out_pjrt, matrix(c(0L, 3L, 4L), nrow = 1L, byrow = TRUE))

  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(f_r(x), out_pjrt)
  expect_identical(as_array(f_quick(x)), out_pjrt)
})

test_that("quickr pipeline matches PJRT: i32 reduce_prod preserves dtype in fast paths", {
  skip_if_no_quickr_or_pjrt()

  check_case <- function(fn, templates, args, expected) {
    graph <- trace_fn(fn, templates)
    out_pjrt <- do.call(quickr_eval_graph_pjrt, c(list(graph), args))
    expect_identical(out_pjrt, expected)

    f_r <- graph_to_quickr_r_function(graph)
    f_quick <- graph_to_quickr_function(graph)

    expect_identical(do.call(f_r, unname(args)), out_pjrt)
    expect_identical(as_array(do.call(f_quick, unname(args))), out_pjrt)
  }

  check_case(
    function(v) {
      nv_reduce_prod(v, dims = 1L, drop = TRUE)
    },
    list(v = nv_array(c(0L, 0L, 0L), dtype = "i32", shape = 3L)),
    list(v = c(2L, 3L, 4L)),
    24L
  )

  check_case(
    function(x2) {
      nv_reduce_prod(x2, dims = c(1L, 2L), drop = TRUE)
    },
    list(x2 = nv_fill(0L, shape = c(2L, 2L), dtype = "i32")),
    list(x2 = matrix(c(2L, 3L, 4L, 5L), nrow = 2L, byrow = TRUE)),
    120L
  )

  check_case(
    function(x3) {
      nv_reduce_prod(x3, dims = 1:3, drop = FALSE)
    },
    list(x3 = nv_array(rep(0L, prod(c(1L, 2L, 2L))), shape = c(1L, 2L, 2L), dtype = "i32")),
    list(x3 = array(c(1L, 2L, 3L, 4L), dim = c(1L, 2L, 2L))),
    array(24L, dim = c(1L, 1L, 1L))
  )
})

test_that("quickr pipeline matches PJRT: i32 power keeps integer semantics", {
  skip_if_no_quickr_or_pjrt()

  pow_i32 <- function(x, y) {
    x^y
  }

  graph <- trace_fn(
    pow_i32,
    list(
      x = nv_fill(0L, shape = c(2L, 2L), dtype = "i32"),
      y = nv_fill(0L, shape = c(2L, 2L), dtype = "i32")
    )
  )

  x <- matrix(c(2L, 3L, 4L, 5L), nrow = 2L, byrow = TRUE)
  y <- matrix(c(-1L, 2L, 3L, 0L), nrow = 2L, byrow = TRUE)

  out_pjrt <- quickr_eval_graph_pjrt(graph, x, y)
  expect_identical(out_pjrt, matrix(c(0L, 9L, 64L, 1L), nrow = 2L, byrow = TRUE))

  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(f_r(x, y), out_pjrt)
  expect_identical(as_array(f_quick(x, y)), out_pjrt)
})

test_that("quickr pipeline matches PJRT: fill/iota/reverse/concatenate/convert/broadcast/transpose/reshape", {
  skip_if_no_quickr_or_pjrt()

  shape_ops <- function(s, v1, v2, x_i32, x21_i32, x13_i32, x3, x_pred) {
    f0 <- nv_fill(3.25, shape = integer(), dtype = "f64")
    f1 <- nv_fill(2L, shape = 4L, dtype = "i32")
    f2 <- nv_fill(TRUE, shape = c(2L, 3L), dtype = "pred")

    i1 <- nv_iota(dim = 1L, dtype = "i32", shape = 5L, start = 1L)
    i2 <- nv_iota(dim = 2L, dtype = "f64", shape = c(2L, 3L), start = 0L)
    i3_empty <- nv_iota(dim = 2L, dtype = "i32", shape = c(0L, 3L), start = 0L)
    i4_noloop <- nv_iota(dim = 2L, dtype = "i32", shape = c(1L, 4L, 1L), start = 0L)

    r1 <- nv_reverse(i1, dims = 1L)
    r2 <- nv_reverse(x_i32, dims = 2L)
    r3 <- nv_reverse(x3, dims = c(1L, 3L))

    c1 <- nv_concatenate(v1, v2, dimension = 1L)
    c2 <- nv_concatenate(x_i32, x_i32, dimension = 1L)
    c3 <- nv_concatenate(x3, x3, dimension = 2L)
    c4 <- nv_concatenate(x13_i32, x13_i32, dimension = 1L)

    s_i32 <- nv_convert(s, dtype = "i32")
    prim_i32 <- nv_convert(x_pred, dtype = "i32")
    i32_pred <- nv_convert(x_i32, dtype = "pred")
    # Use prim_* to force a convert node even when it's a no-op.
    i32_i32 <- prim_convert(x_i32, dtype = "i32")
    pred_pred <- prim_convert(x_pred, dtype = "pred")
    i32_f64 <- prim_convert(x_i32, dtype = "f64")

    b0 <- nv_broadcast_to(s, shape = c(2L, 1L, 3L))
    b1 <- prim_broadcast_in_dim(x_i32, shape = c(2L, 3L), broadcast_dimensions = c(1L, 2L))
    b2 <- nv_broadcast_to(x21_i32, shape = c(2L, 3L))
    b3 <- prim_broadcast_in_dim(x_i32, shape = c(2L, 3L, 1L), broadcast_dimensions = c(1L, 2L))

    t0 <- nv_transpose(x_i32, permutation = c(2L, 1L))
    t1 <- nv_transpose(x_i32, permutation = c(1L, 2L))

    rs <- nv_reshape(s, shape = c(1L, 1L))
    r2a <- nv_reshape(x_i32, shape = c(3L, 2L))
    r3a <- nv_reshape(x3, shape = c(3L, 4L))

    list(
      fill0 = f0,
      fill1 = f1,
      fill2 = f2,
      iota1 = i1,
      iota2 = i2,
      iota_empty = i3_empty,
      iota_noloop = i4_noloop,
      reverse1 = r1,
      reverse2 = r2,
      reverse3 = r3,
      concatenate1 = c1,
      concatenate2 = c2,
      concatenate3 = c3,
      concatenate_1row = c4,
      convert_s = s_i32,
      convert_pred_i32 = prim_i32,
      convert_i32_pred = i32_pred,
      convert_i32_i32 = i32_i32,
      convert_pred_pred = pred_pred,
      convert_i32_f64 = i32_f64,
      broadcast_scalar = b0,
      broadcast_id = b1,
      broadcast_2x1_to_2x3 = b2,
      broadcast_add_dim = b3,
      transpose = t0,
      transpose_id = t1,
      reshape_scalar_fast = rs,
      reshape_2d = r2a,
      reshape_3d = r3a
    )
  }

  templates <- list(
    s = nv_scalar(0.0, dtype = "f64"),
    v1 = nv_array(rep(0L, 3L), shape = 3L, dtype = "i32"),
    v2 = nv_array(rep(0L, 2L), shape = 2L, dtype = "i32"),
    x_i32 = nv_matrix(0L, nrow = 2L, ncol = 3L, dtype = "i32"),
    x21_i32 = nv_matrix(0L, nrow = 2L, ncol = 1L, dtype = "i32"),
    x13_i32 = nv_matrix(0L, nrow = 1L, ncol = 3L, dtype = "i32"),
    x3 = nv_array(rep(0L, prod(c(2L, 2L, 3L))), shape = c(2L, 2L, 3L), dtype = "i32"),
    x_pred = nv_matrix(FALSE, nrow = 2L, ncol = 3L, dtype = "pred")
  )

  run <- list(
    args = list(
      s = 2.5,
      v1 = c(1L, 2L, 3L),
      v2 = c(4L, 5L),
      x_i32 = matrix(1:6, nrow = 2, byrow = TRUE),
      x21_i32 = matrix(c(1L, 2L), nrow = 2),
      x13_i32 = matrix(c(1L, 2L, 3L), nrow = 1L),
      x3 = array(1:12, dim = c(2L, 2L, 3L)),
      x_pred = matrix(c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE), nrow = 2, byrow = TRUE)
    ),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(shape_ops, templates, list(run))
})

test_that("quickr pipeline matches PJRT: broadcast + iota slice assignments compile at rank 3", {
  skip_if_no_quickr_or_pjrt()

  # Regression test: slice assignments at rank 3 with singleton dims (from
  # broadcast) should compile and match PJRT.
  b <- 4L
  n <- 7L
  m <- 5L

  fn <- function(x21) {
    bx <- nv_broadcast_to(x21, shape = c(b, n, m))
    ii <- nv_iota(dim = 2L, dtype = "i32", shape = c(b, n, m), start = 0L)
    bx + nv_convert(ii, dtype = "f64")
  }

  templates <- list(x21 = nv_array(rep(0, prod(c(b, 1L, m))), shape = c(b, 1L, m), dtype = "f64"))

  run <- list(
    args = list(x21 = array(rnorm(b * m), dim = c(b, 1L, m))),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(fn, templates, list(run))
})

test_that("quickr pipeline matches PJRT: zero-length dims (reverse/concatenate/boolean reductions)", {
  skip_if_no_quickr_or_pjrt()

  # This exercises:
  # - 0-extent *inputs* (compile-time interface)
  # - dynamic_slice with 0 slice sizes (0-extent intermediates)
  # - reverse/concatenate/reduce_any/reduce_all on 0-extent arrays.
  zero_dim_ops <- function(X, empty_cols, empty_rows) {
    one <- nv_scalar(1L, dtype = "i32")
    ds_empty_cols <- prim_dynamic_slice(X, one, one, slice_sizes = c(2L, 0L))
    ds_empty_rows <- prim_dynamic_slice(X, one, one, slice_sizes = c(0L, 3L))

    prim_cols <- nv_reverse(empty_cols, dims = 2L) > 0L

    list(
      ds_empty_cols = ds_empty_cols,
      ds_empty_rows = ds_empty_rows,
      cat_rows = nv_concatenate(empty_rows, X, dimension = 1L),
      cat_cols = nv_concatenate(X, empty_cols, dimension = 2L),
      any_cols = nv_reduce_any(prim_cols, dims = 2L, drop = TRUE),
      all_cols = nv_reduce_all(prim_cols, dims = 2L, drop = TRUE),
      any_rows = nv_reduce_any(empty_rows > 0L, dims = 1L, drop = TRUE),
      all_rows = nv_reduce_all(empty_rows > 0L, dims = 1L, drop = TRUE)
    )
  }

  X <- matrix(1:6, nrow = 2, byrow = TRUE)
  empty_cols <- matrix(integer(), nrow = 2L, ncol = 0L)
  empty_rows <- matrix(integer(), nrow = 0L, ncol = 3L)
  graph <- trace_fn(
    zero_dim_ops,
    list(
      X = nv_matrix(0L, nrow = 2L, ncol = 3L, dtype = "i32"),
      empty_cols = nv_matrix(0L, nrow = 2L, ncol = 0L, dtype = "i32"),
      empty_rows = nv_matrix(0L, nrow = 0L, ncol = 3L, dtype = "i32")
    )
  )

  out <- expect_quickr_matches_pjrt(graph, X, empty_cols, empty_rows)
  pjrt <- out$out_pjrt

  expect_equal(pjrt$cat_rows, X)
  expect_equal(pjrt$cat_cols, X)
  expect_equal(as.logical(pjrt$any_cols), rep.int(FALSE, 2L))
  expect_equal(as.logical(pjrt$all_cols), rep.int(TRUE, 2L))
  expect_equal(as.logical(pjrt$any_rows), rep.int(FALSE, 3L))
  expect_equal(as.logical(pjrt$all_rows), rep.int(TRUE, 3L))
})

test_that("quickr pipeline matches PJRT: zero-length dims for numeric sum/prod reductions", {
  skip_if_no_quickr_or_pjrt()

  reduce_empty_numeric <- function(v_empty, empty_cols, empty_rows) {
    list(
      sum_v = nv_reduce_sum(v_empty, dims = 1L, drop = TRUE),
      prod_v_keep = nv_reduce_prod(v_empty, dims = 1L, drop = FALSE),
      sum_cols = nv_reduce_sum(empty_cols, dims = 2L, drop = TRUE),
      prod_cols = nv_reduce_prod(empty_cols, dims = 2L, drop = TRUE),
      sum_rows = nv_reduce_sum(empty_rows, dims = 1L, drop = TRUE),
      prod_rows = nv_reduce_prod(empty_rows, dims = 1L, drop = TRUE),
      sum_full = nv_reduce_sum(empty_cols, dims = c(1L, 2L), drop = TRUE),
      prod_full = nv_reduce_prod(empty_cols, dims = c(1L, 2L), drop = TRUE)
    )
  }

  v_empty <- integer()
  empty_cols <- matrix(integer(), nrow = 2L, ncol = 0L)
  empty_rows <- matrix(integer(), nrow = 0L, ncol = 3L)
  graph <- trace_fn(
    reduce_empty_numeric,
    list(
      v_empty = nv_array(rep(0L, 0L), shape = 0L, dtype = "i32"),
      empty_cols = nv_matrix(0L, nrow = 2L, ncol = 0L, dtype = "i32"),
      empty_rows = nv_matrix(0L, nrow = 0L, ncol = 3L, dtype = "i32")
    )
  )

  out_pjrt <- quickr_eval_graph_pjrt(graph, v_empty, empty_cols, empty_rows)
  expect_identical(out_pjrt$sum_v, 0L)
  expect_identical(out_pjrt$prod_v_keep, array(1L, dim = 1L))
  expect_identical(out_pjrt$sum_cols, array(c(0L, 0L), dim = 2L))
  expect_identical(out_pjrt$prod_cols, array(c(1L, 1L), dim = 2L))
  expect_identical(out_pjrt$sum_rows, array(c(0L, 0L, 0L), dim = 3L))
  expect_identical(out_pjrt$prod_rows, array(c(1L, 1L, 1L), dim = 3L))
  expect_identical(out_pjrt$sum_full, 0L)
  expect_identical(out_pjrt$prod_full, 1L)

  expect_quickr_matches_pjrt(graph, v_empty, empty_cols, empty_rows)
})

test_that("quickr pipeline matches PJRT: reduction branch coverage", {
  skip_if_no_quickr_or_pjrt()

  reduce_ops <- function(v_i32, v_pred, x_i32, x_pred, x3_i32, x3_pred, v_f64, x_f64, x3_f64) {
    sum_noop <- nv_reduce_sum(v_i32, dims = integer(), drop = FALSE)
    sum_rank1_keepdim <- nv_reduce_sum(v_i32, dims = 1L, drop = FALSE)
    sum_rank2_full_drop <- nv_reduce_sum(x_i32, dims = c(1L, 2L), drop = TRUE)
    sum_rank2_full_keep <- nv_reduce_sum(x_i32, dims = c(1L, 2L), drop = FALSE)

    prod_i32 <- nv_reduce_prod(x_i32, dims = 2L, drop = TRUE)
    prod_rank3_full_keep <- nv_reduce_prod(x3_i32, dims = 1:3, drop = FALSE)

    any_rank1_keepdim <- nv_reduce_any(v_pred, dims = 1L, drop = FALSE)
    any_rank2_keepdim <- nv_reduce_any(x_pred, dims = 2L, drop = FALSE)
    all_rank2_drop <- nv_reduce_all(x_pred, dims = 1L, drop = TRUE)
    all_rank3_full_keepdim <- nv_reduce_all(x3_pred, dims = 1:3, drop = FALSE)

    max_rank1_drop <- nv_reduce_max(v_f64, dims = 1L, drop = TRUE)
    min_rank1_keepdim <- nv_reduce_min(v_f64, dims = 1L, drop = FALSE)
    max_rank2_keepdim <- nv_reduce_max(x_f64, dims = 2L, drop = FALSE)
    max_rank3_full_drop <- nv_reduce_max(x3_f64, dims = 1:3, drop = TRUE)
    min_rank3_full_keep <- nv_reduce_min(x3_f64, dims = 1:3, drop = FALSE)

    list(
      sum_noop = sum_noop,
      sum_rank1_keepdim = sum_rank1_keepdim,
      sum_rank2_full_drop = sum_rank2_full_drop,
      sum_rank2_full_keep = sum_rank2_full_keep,
      prod_i32 = prod_i32,
      prod_rank3_full_keep = prod_rank3_full_keep,
      any_rank1_keepdim = any_rank1_keepdim,
      any_rank2_keepdim = any_rank2_keepdim,
      all_rank2_drop = all_rank2_drop,
      all_rank3_full_keepdim = all_rank3_full_keepdim,
      max_rank1_drop = max_rank1_drop,
      min_rank1_keepdim = min_rank1_keepdim,
      max_rank2_keepdim = max_rank2_keepdim,
      max_rank3_full_drop = max_rank3_full_drop,
      min_rank3_full_keep = min_rank3_full_keep
    )
  }

  templates <- list(
    v_i32 = nv_array(rep(0L, 4L), shape = 4L, dtype = "i32"),
    v_pred = nv_array(rep(FALSE, 4L), shape = 4L, dtype = "pred"),
    x_i32 = nv_matrix(0L, nrow = 2L, ncol = 3L, dtype = "i32"),
    x_pred = nv_matrix(FALSE, nrow = 2L, ncol = 3L, dtype = "pred"),
    x3_i32 = nv_array(rep(0L, prod(c(2L, 2L, 3L))), shape = c(2L, 2L, 3L), dtype = "i32"),
    x3_pred = nv_array(rep(FALSE, prod(c(2L, 2L, 3L))), shape = c(2L, 2L, 3L), dtype = "pred"),
    v_f64 = nv_array(rep(0, 4L), shape = 4L, dtype = "f64"),
    x_f64 = nv_matrix(0, nrow = 2L, ncol = 3L, dtype = "f64"),
    x3_f64 = nv_array(rep(0, prod(c(2L, 2L, 3L))), shape = c(2L, 2L, 3L), dtype = "f64")
  )

  run <- list(
    args = list(
      v_i32 = c(1L, 2L, 3L, 4L),
      v_pred = c(TRUE, FALSE, TRUE, TRUE),
      x_i32 = matrix(1:6, nrow = 2, byrow = TRUE),
      x_pred = matrix(c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE), nrow = 2, byrow = TRUE),
      x3_i32 = array(1:12, dim = c(2L, 2L, 3L)),
      x3_pred = array(rep.int(c(TRUE, FALSE), 6L), dim = c(2L, 2L, 3L)),
      v_f64 = c(-0.5, 0.25, -0.1, 0.75),
      x_f64 = matrix(c(-1, 2, -3, 4, -5, 6), nrow = 2, byrow = TRUE),
      x3_f64 = array(seq(-6, 5, by = 1), dim = c(2L, 2L, 3L))
    ),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(reduce_ops, templates, list(run))
})

test_that("quickr pipeline matches PJRT: indexing ops (slice/update/pad) + gather/scatter", {
  skip_if_no_quickr_or_pjrt()

  idx_ops <- function(x, X, X3, s, r, c, a, b, d, upd_s, upd, upd2, padv, idx_vec, sc_idx_vec) {
    ss0 <- nv_static_slice(s, start_indices = integer(), limit_indices = integer(), strides = integer())
    ss1 <- nv_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 2L)
    ss2 <- nv_static_slice(X, start_indices = c(1L, 2L), limit_indices = c(2L, 3L), strides = c(1L, 1L))
    ss3 <- nv_static_slice(
      X3,
      start_indices = c(2L, 1L, 1L),
      limit_indices = c(3L, 3L, 2L),
      strides = c(1L, 1L, 1L)
    )

    ds0 <- prim_dynamic_slice(s, slice_sizes = integer())
    ds1 <- prim_dynamic_slice(x, s, slice_sizes = 3L)
    ds2 <- prim_dynamic_slice(X, r, c, slice_sizes = c(2L, 2L))
    ds3 <- prim_dynamic_slice(X3, a, b, d, slice_sizes = c(2L, 2L, 1L))

    dus0 <- prim_dynamic_update_slice(s, upd_s)
    dus1 <- prim_dynamic_update_slice(x, upd, s)
    dus2 <- prim_dynamic_update_slice(X, upd2, r, c)

    p0 <- nv_pad(s, padv, edge_padding_low = integer(), edge_padding_high = integer())
    p1 <- nv_pad(x, padv, edge_padding_low = 2L, edge_padding_high = 1L, interior_padding = 0L)
    p2 <- nv_pad(X, padv, edge_padding_low = c(1L, 2L), edge_padding_high = c(0L, 1L))
    p3 <- nv_pad(
      X3,
      padv,
      edge_padding_low = c(1L, 0L, 1L),
      edge_padding_high = c(0L, 1L, 0L),
      interior_padding = c(0L, 1L, 0L)
    )

    # Gather/scatter via user-facing subsetting APIs.
    g <- X3[idx_vec, , ]

    base <- nv_iota(dim = 1L, dtype = "i32", shape = 6L, start = 0L)
    scattered_overwrite <- nv_subset_assign(base, sc_idx_vec, value = upd)

    sc_idx <- nv_reshape(sc_idx_vec, c(2L, 1L))
    scattered_add <- prim_scatter(
      base,
      sc_idx,
      upd,
      update_window_dims = integer(),
      inserted_window_dims = 1L,
      input_batching_dims = integer(),
      scatter_indices_batching_dims = integer(),
      scatter_dims_to_operand_dims = 1L,
      index_vector_dim = 2L,
      update_computation = function(old, new) old + new
    )

    list(
      static0 = ss0,
      static1 = ss1,
      static2 = ss2,
      static3 = ss3,
      dynamic0 = ds0,
      dynamic1 = ds1,
      dynamic2 = ds2,
      dynamic3 = ds3,
      update0 = dus0,
      update1 = dus1,
      update2 = dus2,
      pad0 = p0,
      pad1 = p1,
      pad2 = p2,
      pad3 = p3,
      gather = g,
      scatter_overwrite = scattered_overwrite,
      scatter_add = scattered_add
    )
  }

  templates <- list(
    x = nv_array(rep(0L, 6L), shape = 6L, dtype = "i32"),
    X = nv_matrix(0L, nrow = 2L, ncol = 3L, dtype = "i32"),
    X3 = nv_array(rep(0L, prod(c(3L, 4L, 2L))), shape = c(3L, 4L, 2L), dtype = "i32"),
    s = nv_scalar(0L, dtype = "i32"),
    r = nv_scalar(0L, dtype = "i32"),
    c = nv_scalar(0L, dtype = "i32"),
    a = nv_scalar(0L, dtype = "i32"),
    b = nv_scalar(0L, dtype = "i32"),
    d = nv_scalar(0L, dtype = "i32"),
    upd_s = nv_scalar(0L, dtype = "i32"),
    upd = nv_array(rep(0L, 2L), shape = 2L, dtype = "i32"),
    upd2 = nv_matrix(0L, nrow = 2L, ncol = 2L, dtype = "i32"),
    padv = nv_scalar(0L, dtype = "i32"),
    idx_vec = nv_array(rep(0L, 2L), shape = 2L, dtype = "i32"),
    sc_idx_vec = nv_array(rep(0L, 2L), shape = 2L, dtype = "i32")
  )

  X3 <- array(sample.int(100, 24, replace = TRUE), dim = c(3L, 4L, 2L))

  common_args <- list(
    x = 1:6,
    X = matrix(1:6, nrow = 2, byrow = TRUE),
    X3 = X3,
    r = 2L,
    c = 2L,
    a = 0L,
    b = 100L,
    d = 2L,
    upd_s = 11L,
    upd = c(10L, 20L),
    upd2 = matrix(c(1L, 2L, 3L, 4L), nrow = 2, byrow = TRUE),
    padv = 0L,
    idx_vec = c(0L, 4L),
    sc_idx_vec = c(2L, 6L)
  )

  run_clamp_low <- list(
    args = c(common_args, list(s = -100L)),
    info = "clamp_low"
  )

  run_clamp_high <- list(
    args = c(common_args, list(s = 100L)),
    info = "clamp_high"
  )

  expect_quickr_matches_pjrt_fn(idx_ops, templates, list(run_clamp_low, run_clamp_high))
})

test_that("quickr pipeline matches PJRT: gather can return a scalar", {
  skip_if_no_quickr_or_pjrt()

  gather_scalar <- function(x, idx) {
    prim_gather(
      x,
      idx,
      slice_sizes = 1L,
      offset_dims = integer(),
      collapsed_slice_dims = 1L,
      operand_batching_dims = integer(),
      start_indices_batching_dims = integer(),
      start_index_map = 1L,
      index_vector_dim = 1L
    )
  }

  templates <- list(
    x = nv_array(rep(0L, 5L), shape = 5L, dtype = "i32"),
    idx = nv_array(rep(0L, 1L), shape = 1L, dtype = "i32")
  )

  run_low <- list(args = list(x = 1:5, idx = 0L), info = "clamp_low")
  run_high <- list(args = list(x = 1:5, idx = 100L), info = "clamp_high")

  expect_quickr_matches_pjrt_fn(gather_scalar, templates, list(run_low, run_high))
})

test_that("quickr pipeline matches PJRT: dot_general variants", {
  skip_if_no_quickr_or_pjrt()

  dot_ops <- function(a, b, a1, b1, a_dot, b_dot, a_mv, b_mv, lhs, rhs, lhs_b, rhs_b) {
    s <- prim_dot_general(
      a,
      b,
      contracting_dims = list(integer(), integer()),
      batching_dims = list(integer(), integer())
    )
    outer <- prim_dot_general(
      a1,
      b1,
      contracting_dims = list(integer(), integer()),
      batching_dims = list(integer(), integer())
    )
    dot <- prim_dot_general(a_dot, b_dot, contracting_dims = list(1L, 1L), batching_dims = list(integer(), integer()))
    mv <- prim_dot_general(a_mv, b_mv, contracting_dims = list(2L, 1L), batching_dims = list(integer(), integer()))
    mm <- prim_dot_general(lhs, rhs, contracting_dims = list(2L, 1L), batching_dims = list(integer(), integer()))
    batch <- prim_dot_general(lhs_b, rhs_b, contracting_dims = list(3L, 2L), batching_dims = list(1L, 1L))

    list(scalar = s, outer = outer, dot = dot, mv = mv, mm = mm, batch = batch)
  }

  templates <- list(
    a = nv_scalar(0.0, dtype = "f64"),
    b = nv_scalar(0.0, dtype = "f64"),
    a1 = nv_array(rep(0L, 2L), shape = 2L, dtype = "i32"),
    b1 = nv_array(rep(0L, 3L), shape = 3L, dtype = "i32"),
    a_dot = nv_array(rep(0L, 3L), shape = 3L, dtype = "i32"),
    b_dot = nv_array(rep(0L, 3L), shape = 3L, dtype = "i32"),
    a_mv = nv_matrix(0L, nrow = 2L, ncol = 3L, dtype = "i32"),
    b_mv = nv_array(rep(0L, 3L), shape = 3L, dtype = "i32"),
    lhs = nv_matrix(0, nrow = 2L, ncol = 3L, dtype = "f64"),
    rhs = nv_matrix(0, nrow = 3L, ncol = 4L, dtype = "f64"),
    lhs_b = nv_array(rep(0, prod(c(2L, 3L, 4L))), shape = c(2L, 3L, 4L), dtype = "f64"),
    rhs_b = nv_array(rep(0, prod(c(2L, 4L, 5L))), shape = c(2L, 4L, 5L), dtype = "f64")
  )

  run <- list(
    args = list(
      a = 2,
      b = 3,
      a1 = 1:2,
      b1 = c(10L, 20L, 30L),
      a_dot = 1:3,
      b_dot = 4:6,
      a_mv = matrix(1:6, nrow = 2, byrow = TRUE),
      b_mv = 1:3,
      lhs = matrix(c(0.2, -0.1, 0.3, 0.4, 0.5, -0.2), nrow = 2, byrow = TRUE),
      rhs = matrix(c(0.1, 0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, 0.9, -1.0, 1.1, -1.2), nrow = 3, byrow = TRUE),
      lhs_b = array(0.1 * (1:24), dim = c(2L, 3L, 4L)),
      rhs_b = array(0.05 * (1:40), dim = c(2L, 4L, 5L))
    ),
    info = "run"
  )

  expect_quickr_matches_pjrt_fn(dot_ops, templates, list(run))
})

test_that("quickr pipeline matches PJRT: control flow (if/while)", {
  skip_if_no_quickr_or_pjrt()

  cf_ops <- function(p) {
    if_s <- nv_if(p, \() 1L, \() 2L)
    if_a <- nv_if(
      p,
      \() nv_fill(1L, shape = c(2L, 2L), dtype = "i32"),
      \() nv_fill(2L, shape = c(2L, 2L), dtype = "i32")
    )

    w1 <- nv_while(
      init = list(i = 0L, total = 0L),
      cond = function(i, total) i < 5L,
      body = function(i, total) list(i = i + 1L, total = total + i)
    )

    w2 <- nv_while(
      init = list(i = 0L, v = nv_iota(dim = 1L, dtype = "i32", shape = 5L, start = 0L)),
      cond = function(i, v) i < 3L,
      body = function(i, v) list(i = i + 1L, v = v + 2L)
    )

    list(if_s = if_s, if_a = if_a, w1 = w1, w2 = w2)
  }

  templates <- list(p = nv_scalar(FALSE, dtype = "pred"))

  run_true <- list(args = list(p = TRUE), info = "p=TRUE")
  run_false <- list(args = list(p = FALSE), info = "p=FALSE")

  expect_quickr_matches_pjrt_fn(cf_ops, templates, list(run_true, run_false))
})
