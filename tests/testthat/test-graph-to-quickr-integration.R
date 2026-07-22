test_that("integration: MNIST-shaped classifier training from rank-5 image batch via quickr loss + grad", {
  skip_if_no_quickr_or_pjrt()

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
    logits <- nv_matmul(X2, W) + nv_broadcast_to(b, shape = c(n, k))
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

  f_quick <- graph_to_quickr_function(graph, unwrap = TRUE)
  f_r <- graph_to_quickr_r_function(graph)
  run_pjrt <- graph_runner_pjrt(graph)

  out_r0 <- f_r(X, Y, W, b)
  out_quick0 <- f_quick(X, Y, W, b)
  out_pjrt0 <- run_pjrt(X, Y, W, b)
  expect_equal(out_quick0, out_pjrt0, tolerance = 1e-10)
  expect_equal(out_r0, out_pjrt0, tolerance = 1e-10)

  lr <- 0.1
  losses <- numeric(5)
  for (iter in seq_along(losses)) {
    out_pjrt <- run_pjrt(X, Y, W, b)
    out_quick <- f_quick(X, Y, W, b)
    expect_equal(out_quick, out_pjrt, tolerance = 1e-10)

    losses[[iter]] <- as.numeric(out_quick$loss)
    W <- W - lr * out_quick$grad$W
    b <- b - lr * out_quick$grad$b
  }
  expect_lt(losses[[length(losses)]], losses[[1L]])
})

test_that("integration: tfp/greta-like log_prob + grad workflow via quickr", {
  skip_if_no_quickr_or_pjrt()

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

  f_quick <- graph_to_quickr_function(graph, unwrap = TRUE)
  f_r <- graph_to_quickr_r_function(graph)
  run_pjrt <- graph_runner_pjrt(graph)

  out_r0 <- f_r(0.1, -0.2)
  out_pjrt0 <- run_pjrt(0.1, -0.2)
  out_quick0 <- f_quick(0.1, -0.2)
  expect_equal(out_quick0, out_pjrt0, tolerance = 1e-10)
  expect_equal(out_r0, out_pjrt0, tolerance = 1e-10)

  # A few steps of gradient ascent (MAP for this quadratic objective).
  w_quick <- 0.0
  b_quick <- 0.0
  w_pjrt <- 0.0
  b_pjrt <- 0.0

  out0 <- f_quick(w_quick, b_quick)
  lp0 <- as.numeric(out0$log_prob)

  step <- 0.01
  for (iter in seq_len(10)) {
    out_pjrt <- run_pjrt(w_pjrt, b_pjrt)
    out_quick <- f_quick(w_quick, b_quick)

    expect_equal(out_quick$log_prob, out_pjrt$log_prob, tolerance = 1e-12)
    expect_equal(out_quick$grad$w, out_pjrt$grad$w, tolerance = 1e-12)
    expect_equal(out_quick$grad$b, out_pjrt$grad$b, tolerance = 1e-12)

    w_quick <- w_quick + step * as.numeric(out_quick$grad$w)
    b_quick <- b_quick + step * as.numeric(out_quick$grad$b)
    w_pjrt <- w_pjrt + step * as.numeric(out_pjrt$grad$w)
    b_pjrt <- b_pjrt + step * as.numeric(out_pjrt$grad$b)

    expect_equal(w_quick, w_pjrt, tolerance = 1e-12)
    expect_equal(b_quick, b_pjrt, tolerance = 1e-12)
  }

  final_quick <- f_quick(w_quick, b_quick)
  lp1_quick <- as.numeric(final_quick$log_prob)
  lp1_pjrt <- as.numeric(run_pjrt(w_pjrt, b_pjrt)$log_prob)
  expect_equal(lp1_quick, lp1_pjrt, tolerance = 1e-12)
  expect_gt(lp1_quick, lp0)
})
