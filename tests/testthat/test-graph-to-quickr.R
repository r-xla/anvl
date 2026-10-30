expect_lowering_error <- function(fn, templates, pattern) {
  graph <- trace_fn(fn, templates)
  testthat::expect_error(graph_to_r_function(graph), pattern, fixed = FALSE)
}

test_that("the lowered function supports list outputs", {
  graph <- trace_fn(
    function(x) {
      list(a = x, b = x + x)
    },
    list(x = nv_scalar(1.0, dtype = "f64"))
  )

  expect_r_matches_pjrt(graph, 0.5)
})

test_that("graph_to_quickr_r_function lowers a graph to a plain R function", {
  graph <- trace_fn(
    function(x) x + 1L,
    list(x = nv_scalar(1.0, dtype = "f64")),
    desc = local_descriptor()
  )
  expect_equal(graph_to_quickr_r_function(graph)(2), 3)
})

test_that("graph_to_quickr_function rejects non-graph inputs", {
  expect_error(graph_to_quickr_function(1), "must be a .*AnvlGraph", fixed = FALSE)
})

test_that("graph_to_quickr_r_function rejects non-graph inputs", {
  expect_error(graph_to_quickr_r_function(1), "must be a .*AnvlGraph", fixed = FALSE)
})

test_that("the lowered function supports nested inputs", {
  graph <- trace_fn(
    function(x) {
      x$a + x$b
    },
    list(
      x = list(
        a = nv_scalar(1.0, dtype = "f64"),
        b = nv_scalar(2.0, dtype = "f64")
      )
    )
  )

  expect_r_matches_pjrt(graph, x = list(a = 0.5, b = 1.25))
})

test_that("the lowered function handles GraphLiteral inputs (R scalar literals)", {
  graph <- trace_fn(
    function(x) {
      x + 1L
    },
    list(x = nv_scalar(0.0, dtype = "f64")),
    desc = local_descriptor()
  )

  expect_r_matches_pjrt(graph, 2.5)
})

test_that("the lowered function produces a stable flat signature", {
  graph <- trace_fn(
    function(out, v1) {
      out + v1
    },
    list(
      out = nv_scalar(1.0, dtype = "f64"),
      v1 = nv_scalar(2.0, dtype = "f64")
    )
  )

  f_wrapped <- graph_to_r_function(graph)
  f_r <- graph_to_quickr_r_function(graph)
  expect_identical(names(formals(f_wrapped)), c("x1", "x2"))
  expect_identical(names(formals(f_r)), c("x1", "x2"))

  expect_r_matches_pjrt(graph, 0.5, 1.25)
})

test_that("the lowered function handles zero-length output leaves", {
  graph <- trace_fn(
    function() {
      list(
        b = nv_fill(2.0, shape = integer(), dtype = "f64"),
        a = nv_fill(1.0, shape = c(0L, 2L), dtype = "f64"),
        c = nv_fill(3.0, shape = integer(), dtype = "f64")
      )
    },
    list()
  )

  out <- expect_r_matches_pjrt(graph)
  expect_identical(dim(out$out_r$a), c(0L, 2L))
})

test_that("the lowered function ignores static args when flattening nested inputs", {
  # Regression test: when tracing with nested list inputs and an extra static arg,
  # graph$inputs contains only the array leaves but wrapper flattening sees both.
  graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x$a$u + x$b
      } else {
        x$a$v + x$b
      }
    },
    list(
      x = list(
        a = list(
          u = nv_scalar(1.0, dtype = "f64"),
          v = nv_scalar(2.0, dtype = "f64")
        ),
        b = nv_scalar(3.0, dtype = "f64")
      ),
      flag = TRUE
    )
  )

  x <- list(a = list(u = 0.5, v = 1.5), b = 2.0)
  expect_r_matches_pjrt(graph, x = x, flag = TRUE)
})

test_that("the lowered function accepts flat static args (no nested inputs)", {
  fn <- function(x, flag) {
    if (flag) {
      x + 1L
    } else {
      x - 1L
    }
  }

  graph <- trace_fn(
    fn,
    list(
      x = nv_scalar(0.0, dtype = "f64"),
      flag = TRUE
    ),
    desc = local_descriptor()
  )

  f_wrapped <- graph_to_r_function(graph)
  f_r <- graph_to_quickr_r_function(graph)
  expect_identical(names(formals(f_wrapped)), c("x", "flag"))
  expect_identical(names(formals(f_r)), c("x", "flag"))

  out_true <- expect_r_matches_pjrt(graph, x = 0.5, flag = TRUE)
  expect_equal(out_true$out_r, 1.5)
})

test_that("the lowered function's wrappers reject mismatched runtime static args", {
  flat_graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x + 1L
      } else {
        x - 1L
      }
    },
    list(
      x = nv_scalar(0.0, dtype = "f64"),
      flag = TRUE
    ),
    desc = local_descriptor()
  )

  flat_r <- graph_to_quickr_r_function(flat_graph)
  flat_wrapped <- graph_to_r_function(flat_graph)

  expect_error(flat_r(0.5, FALSE), "Static arguments must match", fixed = FALSE)
  expect_error(flat_wrapped(0.5, FALSE), "Static arguments must match", fixed = FALSE)

  nested_graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x$a$u + x$b
      } else {
        x$a$v + x$b
      }
    },
    list(
      x = list(
        a = list(
          u = nv_scalar(1.0, dtype = "f64"),
          v = nv_scalar(2.0, dtype = "f64")
        ),
        b = nv_scalar(3.0, dtype = "f64")
      ),
      flag = TRUE
    )
  )

  nested_r <- graph_to_quickr_r_function(nested_graph)
  nested_wrapped <- graph_to_r_function(nested_graph)
  x <- list(a = list(u = 0.5, v = 1.5), b = 2.0)

  expect_error(nested_r(x = x, flag = FALSE), "Static arguments must match", fixed = FALSE)
  expect_error(nested_wrapped(x = x, flag = FALSE), "Static arguments must match", fixed = FALSE)
})

test_that("the lowered function generates placeholder names for unnamed `...` inputs", {
  graph <- trace_fn(
    function(...) {
      xs <- list(...)
      xs[[1L]] + xs[[2L]]$a + xs[[2L]]$b
    },
    list(
      nv_scalar(0.0, dtype = "f64"),
      list(
        a = nv_scalar(1.0, dtype = "f64"),
        b = nv_scalar(2.0, dtype = "f64")
      )
    )
  )

  # Formals are generated from empty in_tree names; call by position to keep this robust.
  out <- expect_r_matches_pjrt(graph, 0.25, list(a = 0.5, b = 1.0))
  expect_equal(out$out_r, 0.25 + 0.5 + 1.0)
})

test_that("the lowered function errors on mismatched flattened input structure", {
  graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x$a$u + x$b
      } else {
        x$a$v + x$b
      }
    },
    list(
      x = list(
        a = list(
          u = nv_scalar(1.0, dtype = "f64"),
          v = nv_scalar(2.0, dtype = "f64")
        ),
        b = nv_scalar(3.0, dtype = "f64")
      ),
      flag = TRUE
    )
  )

  f_wrapped <- graph_to_r_function(graph)
  f_r <- graph_to_quickr_r_function(graph)
  x_bad <- list(a = list(u = 0.5), b = 2.0)
  expect_error(
    f_wrapped(x = x_bad, flag = TRUE),
    "Expected 4 flattened inputs, got 3",
    fixed = FALSE
  )
  expect_error(
    f_r(x = x_bad, flag = TRUE),
    "Expected 4 flattened inputs, got 3",
    fixed = FALSE
  )
})

test_that("the lowered function preserves 1D output axes", {
  graph <- trace_fn(
    function(x) {
      x + 1L
    },
    list(x = nv_array(numeric(3), shape = 3L, dtype = "f64")),
    desc = local_descriptor()
  )

  out <- expect_r_matches_pjrt(graph, c(0.5, 1.5, 2.5))
  expect_identical(dim(out$out_r), c(3L))
})

test_that("the lowered function preserves empty 1D output axes without wrapper", {
  graph <- trace_fn(
    function(x) {
      x
    },
    list(x = nv_array(numeric(0), shape = 0L, dtype = "f64"))
  )

  x <- array(numeric(), dim = 0L)
  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  f_r <- graph_to_quickr_r_function(graph)
  f_wrapped <- graph_to_r_function(graph)

  expect_identical(dim(out_pjrt), 0L)
  expect_identical(dim(f_r(x)), 0L)
  expect_identical(dim(f_wrapped(x)), 0L)
  expect_identical(f_r(x), out_pjrt)
  expect_identical(as_array(f_wrapped(x)), out_pjrt)
})

test_that("the lowered function preserves rank-1 axes for direct outputs", {
  graph <- trace_fn(
    function(x) {
      nv_expm1(x)
    },
    list(x = nv_array(numeric(3), shape = 3L, dtype = "f64"))
  )

  x <- array(c(0.1, 0.2, 0.3), dim = 3L)
  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  f_r <- graph_to_quickr_r_function(graph)
  f_wrapped <- graph_to_r_function(graph)

  expect_identical(dim(out_pjrt), c(3L))
  expect_identical(dim(f_r(x)), c(3L))
  expect_identical(dim(f_wrapped(x)), c(3L))
  expect_equal(f_r(x), out_pjrt, tolerance = 1e-12)
  expect_equal(as_array(f_wrapped(x)), out_pjrt, tolerance = 1e-12)
})

test_that("the lowered function preserves rank-1 axes for wrapped outputs", {
  graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        nv_log1p(x)
      } else {
        nv_log1p(x)
      }
    },
    list(
      x = nv_array(numeric(3), shape = 3L, dtype = "f64"),
      flag = TRUE
    )
  )

  x <- array(c(0.1, 0.2, 0.3), dim = 3L)
  out_pjrt <- quickr_eval_graph_pjrt(graph, x, TRUE)
  f_r <- graph_to_quickr_r_function(graph)
  f_wrapped <- graph_to_r_function(graph)

  expect_identical(dim(out_pjrt), c(3L))
  expect_identical(dim(f_r(x, TRUE)), c(3L))
  expect_identical(dim(f_wrapped(x, TRUE)), c(3L))
  expect_equal(f_r(x, TRUE), out_pjrt, tolerance = 1e-12)
  expect_equal(as_array(f_wrapped(x, TRUE)), out_pjrt, tolerance = 1e-12)
})

test_that("the lowered function preserves pred leaves in multiple outputs", {
  graph <- trace_fn(
    function(x) {
      list(p = x > 0L, v = x + 1L)
    },
    list(x = nv_scalar(0.0, dtype = "f64")),
    desc = local_descriptor()
  )

  out <- expect_r_matches_pjrt(graph, -0.5)
  out <- out$out_r

  expect_identical(out$p, FALSE)
  expect_equal(out$v, 0.5)
})

test_that("the lowering errors on unsupported primitives", {
  unsupported_fns <- list(
    function(x) nv_popcnt(x),
    function(x, y) nv_shift_left(x, y),
    function(x, y) nv_shift_right_logical(x, y),
    function(x, y) nv_shift_right_arithmetic(x, y)
  )

  templates_1 <- list(x = nv_array(1L, dtype = "i32", shape = 1L))
  templates_2 <- list(
    x = nv_array(1L, dtype = "i32", shape = 1L),
    y = nv_array(1L, dtype = "i32", shape = 1L)
  )

  for (fn in unsupported_fns) {
    templates <- if (length(formals(fn)) == 1L) templates_1 else templates_2
    expect_lowering_error(fn, templates, "does not support these primitives")
  }
})

test_that("the lowering validates unsupported primitives in nested subgraphs", {
  graph <- trace_fn(
    function(p, x) {
      nv_if(p, \() nv_popcnt(x), \() x)
    },
    list(
      p = nv_scalar(TRUE, dtype = "pred"),
      x = nv_scalar(1L, dtype = "i32")
    )
  )

  expect_error(
    graph_to_r_function(graph),
    "does not support these primitives: popcnt",
    fixed = FALSE
  )
  expect_error(
    graph_to_quickr_r_function(graph),
    "does not support these primitives: popcnt",
    fixed = FALSE
  )
})

test_that("the lowering errors on unsupported ranks", {
  # input rank > 5 is rejected during quickr argument declaration
  x6 <- array(0, dim = rep(1L, 6L))
  expect_lowering_error(
    function(x) x,
    list(x = nv_array(x6, dtype = "f64", shape = dim(x6))),
    "supports arrays up to rank 5"
  )

  # output rank > 5 in fill is rejected in the emitter
  expect_lowering_error(
    function() nv_fill(1.0, shape = rep(1L, 6L), dtype = "f64"),
    list(),
    "only arrays up to rank 5"
  )
})

test_that("the lowering rejects unsupported dtypes", {
  graph <- trace_fn(function(x) x + x, list(x = nv_aval("i64", c())))
  testthat::expect_error(graph_to_r_function(graph), "Unsupported dtype.*i64", fixed = FALSE)
})

test_that("the lowering rejects transpose ranks other than 2", {
  x3 <- array(1:8, dim = c(2L, 2L, 2L))
  graph <- trace_fn(
    function(x) prim_transpose(x, perm = c(2L, 1L, 3L)),
    list(x = nv_array(x3, dtype = "i32", shape = dim(x3)))
  )
  testthat::expect_error(graph_to_r_function(graph), "transpose: only rank-2", fixed = FALSE)
})

test_that("the lowering rejects reshape ranks > 5", {
  x5 <- array(1, dim = rep(1L, 5L))
  graph <- trace_fn(
    function(x) prim_reshape(x, shape = rep(1L, 6L)),
    list(x = nv_array(x5, dtype = "i32", shape = dim(x5)))
  )
  testthat::expect_error(graph_to_r_function(graph), "reshape: only arrays up to rank 5", fixed = FALSE)
})

test_that("the lowering rejects broadcast_in_axes ranks > 5", {
  graph <- trace_fn(
    function(x) prim_broadcast_in_axes(x, shape = rep(1L, 6L), broadcast_axes = 6L),
    list(x = nv_array(1L, dtype = "i32", shape = 1L))
  )
  expect_error(graph_to_r_function(graph), "broadcast_in_axes: only arrays up to rank 5", fixed = FALSE)
})

test_that("the lowering rejects reductions over empty axes", {
  templ <- list(x = nv_aval("f64", c(2L, 0L)))
  graph <- trace_fn(function(x) prim_max(x, axes = 2L, drop = TRUE), templ)
  expect_error(graph_to_r_function(graph), "empty axes", fixed = FALSE)
})

test_that("the lowering rejects unsupported reduce_sum variants", {
  # The lowering's other guards (scalar / rank-1 / rank-2 with out-of-range
  # axes) are unreachable through `prim_sum()`, which validates `axes`
  # against the input rank at trace time -- see test-primitives-stablehlo.R.
  graph <- trace_fn(
    function(x) prim_sum(x, axes = 2L, drop = TRUE),
    list(x = nv_array(1:8, shape = c(2L, 2L, 2L), dtype = "i32"))
  )
  expect_error(graph_to_r_function(graph), "for rank > 2, only full reductions", fixed = FALSE)
})

test_that("the lowering rejects an i8 input", {
  graph <- trace_fn(
    function(x) x,
    list(x = nv_aval("i8", 1L))
  )

  expect_error(graph_to_r_function(graph), "Unsupported dtype.*i8")
})

test_that("integration: MNIST-shaped classifier training from rank-5 image batch via loss + grad", {
  withr::local_seed(1)

  n <- 16L
  h <- 28L
  w <- 28L
  c <- 1L
  t <- 1L
  d <- h * w * c * t
  k <- 10L

  X <- array(rnorm(n * h * w * c * t, sd = 0.5), dim = c(n, h, w, c, t))
  labels <- sample.int(k, size = n, replace = TRUE)
  Y <- matrix(0, nrow = n, ncol = k)
  Y[cbind(seq_len(n), labels)] <- 1

  W <- matrix(rnorm(d * k, sd = 0.05), nrow = d, ncol = k)
  b <- rnorm(k, sd = 0.05)

  scale <- nv_scalar(1.0 / (n * k), dtype = "f64")

  loss_fn <- function(X, Y, W, b) {
    X2 <- nv_reshape(X, shape = c(n, d))
    logits <- nv_matmul(X2, W) + prim_broadcast_in_axes(b, shape = c(n, k), broadcast_axes = 2L)
    resid <- logits - Y
    sum(resid * resid) * scale
  }

  loss_and_grad <- function(X, Y, W, b) {
    g <- gradient(loss_fn, wrt = c("W", "b", "X"))
    list(
      loss = loss_fn(X, Y, W, b),
      grad = g(X, Y, W, b)
    )
  }

  graph <- expect_no_warning(trace_fn(
    loss_and_grad,
    list(
      X = nv_array(X, dtype = "f64", shape = dim(X)),
      Y = nv_array(Y, dtype = "f64", shape = dim(Y)),
      W = nv_array(W, dtype = "f64", shape = dim(W)),
      b = nv_array(b, dtype = "f64", shape = c(k))
    )
  ))

  f_r <- graph_to_quickr_r_function(graph)
  run_pjrt <- graph_runner_pjrt(graph)

  out_r0 <- f_r(X, Y, W, b)
  out_pjrt0 <- run_pjrt(X, Y, W, b)
  expect_equal(out_r0, out_pjrt0, tolerance = 1e-10)

  lr <- 0.1
  losses <- numeric(5)
  for (iter in seq_along(losses)) {
    out_pjrt <- run_pjrt(X, Y, W, b)
    out_r <- f_r(X, Y, W, b)
    expect_equal(out_r, out_pjrt, tolerance = 1e-10)

    losses[[iter]] <- as.numeric(out_r$loss)
    W <- W - lr * out_r$grad$W
    b <- b - lr * out_r$grad$b
  }
  expect_lt(losses[[length(losses)]], losses[[1L]])
})

test_that("integration: tfp/greta-like log_prob + grad workflow", {
  withr::local_seed(2)

  n <- 20L
  x <- as.numeric(scale(seq_len(n)))
  y <- 1.5 * x - 0.3 + rnorm(n, sd = 0.15)

  x_nv <- nv_array(x, dtype = "f64", shape = c(n))
  y_nv <- nv_array(y, dtype = "f64", shape = c(n))

  half <- nv_scalar(0.5, dtype = "f64")

  log_joint <- function(w, b) {
    mu <- w * x_nv + b
    resid <- y_nv - mu
    ll <- -half * sum(resid * resid)
    prior <- -half * (w * w + b * b)
    ll + prior
  }

  logp_and_grad <- function(w, b) {
    g <- gradient(log_joint, wrt = c("w", "b"))
    list(
      log_prob = log_joint(w, b),
      grad = g(w, b)
    )
  }

  graph <- expect_no_warning(trace_fn(
    logp_and_grad,
    list(
      w = nv_scalar(0.0, dtype = "f64"),
      b = nv_scalar(0.0, dtype = "f64")
    )
  ))

  f_r <- graph_to_quickr_r_function(graph)
  run_pjrt <- graph_runner_pjrt(graph)

  out_r0 <- f_r(0.1, -0.2)
  out_pjrt0 <- run_pjrt(0.1, -0.2)
  expect_equal(out_r0, out_pjrt0, tolerance = 1e-10)

  # A few steps of gradient ascent (MAP for this quadratic objective).
  w_r <- 0.0
  b_r <- 0.0
  w_pjrt <- 0.0
  b_pjrt <- 0.0

  out0 <- f_r(w_r, b_r)
  lp0 <- as.numeric(out0$log_prob)

  step <- 0.01
  for (iter in seq_len(10)) {
    out_pjrt <- run_pjrt(w_pjrt, b_pjrt)
    out_r <- f_r(w_r, b_r)

    expect_equal(out_r$log_prob, out_pjrt$log_prob, tolerance = 1e-12)
    expect_equal(out_r$grad$w, out_pjrt$grad$w, tolerance = 1e-12)
    expect_equal(out_r$grad$b, out_pjrt$grad$b, tolerance = 1e-12)

    w_r <- w_r + step * as.numeric(out_r$grad$w)
    b_r <- b_r + step * as.numeric(out_r$grad$b)
    w_pjrt <- w_pjrt + step * as.numeric(out_pjrt$grad$w)
    b_pjrt <- b_pjrt + step * as.numeric(out_pjrt$grad$b)

    expect_equal(w_r, w_pjrt, tolerance = 1e-12)
    expect_equal(b_r, b_pjrt, tolerance = 1e-12)
  }

  final_r <- f_r(w_r, b_r)
  lp1_r <- as.numeric(final_r$log_prob)
  lp1_pjrt <- as.numeric(run_pjrt(w_pjrt, b_pjrt)$log_prob)
  expect_equal(lp1_r, lp1_pjrt, tolerance = 1e-12)
  expect_gt(lp1_r, lp0)
})

test_that("integration: indexing-heavy graph matches PJRT", {
  x <- array(sample.int(50, 24, replace = TRUE), dim = c(3L, 4L, 2L))
  idx <- matrix(c(0L, 4L), ncol = 1L) # gather clamps out-of-bounds
  padv <- 0L
  s1 <- 0L
  s2 <- 100L
  sc_idx <- matrix(c(2L, 6L), ncol = 1L)

  fn <- function(x, idx, padv, s1, s2, sc_idx) {
    g <- prim_gather(
      x,
      idx,
      slice_sizes = c(1L, 2L, 2L),
      offset_axes = c(2L, 3L),
      collapsed_slice_axes = 1L,
      x_batching_axes = integer(),
      start_indices_batching_axes = integer(),
      start_index_map = 1L,
      index_vector_axis = 2L
    )

    r <- nv_rev(g, axes = 3L)

    p <- prim_pad(
      r,
      padv,
      edge_padding_low = c(1L, 0L, 0L),
      edge_padding_high = c(0L, 1L, 0L),
      interior_padding = c(0L, 1L, 0L)
    )

    s <- prim_dynamic_slice(p, s1, s2, 1L, slice_sizes = c(2L, 2L, 2L))
    m <- prim_reshape(s, shape = c(2L, 4L))
    upd <- prim_sum(m, axes = 2L, drop = TRUE)

    base <- nv_iota(axis = 1L, shape = 6L, dtype = "i32", start = 0L)
    scattered <- prim_scatter(
      base,
      sc_idx,
      upd,
      update_window_axes = integer(),
      inserted_window_axes = 1L,
      x_batching_axes = integer(),
      scatter_indices_batching_axes = integer(),
      scatter_axes_to_x_axes = 1L,
      index_vector_axis = 2L,
      update_fn = function(old, new) old + new
    )

    out_f64 <- prim_convert(scattered, dtype = "f64")
    list(slice = s, scattered = scattered, total = sum(out_f64))
  }

  templates <- list(
    x = nv_array(x, dtype = "i32", shape = dim(x)),
    idx = nv_array(idx, dtype = "i32", shape = dim(idx)),
    padv = nv_scalar(0L, dtype = "i32"),
    s1 = nv_scalar(0L, dtype = "i32"),
    s2 = nv_scalar(0L, dtype = "i32"),
    sc_idx = nv_array(sc_idx, dtype = "i32", shape = dim(sc_idx))
  )

  run <- list(
    args = list(
      x = x,
      idx = idx,
      padv = padv,
      s1 = s1,
      s2 = s2,
      sc_idx = sc_idx
    ),
    info = "run"
  )

  expect_r_matches_pjrt_fn(fn, templates, list(run))
})

test_that("integration: `%*%` matches PJRT for array ranks 1..5", {
  withr::local_seed(4)

  m <- 3L
  n <- 4L
  p <- 5L
  batch_dim <- 2L

  matmul_any_rank <- function(A, B) {
    if (naxes(A) == 1L) {
      A <- nv_reshape(A, c(1L, shape(A)))
    }
    if (naxes(B) == 1L) {
      B <- nv_reshape(B, c(shape(B), 1L))
    }

    pad_and_broadcast <- function(x, target_shape) {
      if (identical(shape(x), target_shape)) {
        return(x)
      }
      if (naxes(x) < length(target_shape)) {
        x <- nv_reshape(x, c(rep.int(1L, length(target_shape) - naxes(x)), shape(x)))
      }
      prim_broadcast_in_axes(x, shape = target_shape, broadcast_axes = seq_along(target_shape))
    }

    shA <- shape(A)
    shB <- shape(B)
    rA <- length(shA)
    rB <- length(shB)
    target_batch <- if (rA >= rB) {
      shA[seq_len(rA - 2L)]
    } else {
      shB[seq_len(rB - 2L)]
    }

    A <- pad_and_broadcast(A, c(target_batch, shA[(rA - 1L):rA]))
    B <- pad_and_broadcast(B, c(target_batch, shB[(rB - 1L):rB]))

    A %*% B
  }

  template_array <- function(x) {
    shp <- dim(x)
    if (is.null(shp)) {
      shp <- c(length(x))
    }
    nv_array(x, dtype = "f64", shape = shp)
  }

  make_input <- function(rank, side) {
    if (rank == 1L) {
      return(rnorm(n, sd = 0.2))
    }
    leading <- rep.int(batch_dim, rank - 2L)
    if (side == "lhs") {
      dims <- c(leading, m, n)
    } else {
      dims <- c(leading, n, p)
    }
    array(rnorm(prod(dims), sd = 0.2), dim = dims)
  }

  rank_pairs <- list(
    c(1L, 1L),
    c(1L, 2L),
    c(2L, 1L),
    c(2L, 2L),
    c(3L, 5L),
    c(5L, 3L)
  )

  for (pair in rank_pairs) {
    a_rank <- pair[[1L]]
    b_rank <- pair[[2L]]

    A <- make_input(a_rank, "lhs")
    B <- make_input(b_rank, "rhs")

    templates <- list(A = template_array(A), B = template_array(B))
    run <- list(
      args = list(A = A, B = B),
      info = paste0("a_rank=", a_rank, ", b_rank=", b_rank)
    )
    expect_r_matches_pjrt_fn(matmul_any_rank, templates, list(run), tolerance = 1e-12)
  }
})

test_that("the lowering handles a long scalar chain (stress)", {
  decay <- nv_scalar(0.999, dtype = "f64")
  shift <- nv_scalar(0.001, dtype = "f64")

  # Stress the lowering with a longer linear chain.
  n_steps <- 8L
  chain_fn <- function(x) {
    for (i in seq_len(n_steps)) {
      x <- x * decay + shift
    }
    x
  }

  x <- 0.123
  templates <- list(x = nv_scalar(0.0, dtype = "f64"))
  run <- list(args = list(x = x), info = "run")
  expect_r_matches_pjrt_fn(chain_fn, templates, list(run), tolerance = 1e-12)
})
