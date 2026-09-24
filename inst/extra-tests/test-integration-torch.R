source(system.file("extra-tests", "torch-helpers.R", package = "anvl"))

test_that("MLP forward and reverse", {
  relu <- function(x) (x + abs(x)) / 2

  mlp <- function(x, W1, b1, W2, b2, matmul) {
    h1 <- matmul(W1, x) + b1
    h1_relu <- relu(h1)
    matmul(W2, h1_relu) + b2
  }

  mse_loss <- function(pred, y) mean((pred - y)^2)

  mlp_loss <- function(x, y, W1, b1, W2, b2, matmul) {
    pred <- mlp(x, W1, b1, W2, b2, matmul)
    mse_loss(pred, y)
  }

  matmul_nv <- function(a, b) a %*% b
  matmul_torch <- function(a, b) a$matmul(b)

  # Input axis: 4, hidden axis: 6, output axis: 2 (x, y, biases are column vectors)
  x_data <- array(rnorm(4), dim = c(4L, 1L))
  y_data <- array(rnorm(2), dim = c(2L, 1L))
  W1_data <- array(rnorm(24), dim = c(6L, 4L))
  b1_data <- array(rnorm(6), dim = c(6L, 1L))
  W2_data <- array(rnorm(12), dim = c(2L, 6L))
  b2_data <- array(rnorm(2), dim = c(2L, 1L))

  x_nv <- nv_array(x_data, dtype = "f32")
  y_nv <- nv_array(y_data, dtype = "f32")
  W1_nv <- nv_array(W1_data, dtype = "f32")
  b1_nv <- nv_array(b1_data, dtype = "f32")
  W2_nv <- nv_array(W2_data, dtype = "f32")
  b2_nv <- nv_array(b2_data, dtype = "f32")

  x_torch <- torch::torch_tensor(x_data, dtype = torch::torch_float32(), requires_grad = TRUE)
  y_torch <- torch::torch_tensor(y_data, dtype = torch::torch_float32())
  W1_torch <- torch::torch_tensor(W1_data, dtype = torch::torch_float32(), requires_grad = TRUE)
  b1_torch <- torch::torch_tensor(b1_data, dtype = torch::torch_float32(), requires_grad = TRUE)
  W2_torch <- torch::torch_tensor(W2_data, dtype = torch::torch_float32(), requires_grad = TRUE)
  b2_torch <- torch::torch_tensor(b2_data, dtype = torch::torch_float32(), requires_grad = TRUE)

  # Forward and reverse pass with value_and_gradient
  mlp_loss_nv <- function(x, y, W1, b1, W2, b2) mlp_loss(x, y, W1, b1, W2, b2, matmul_nv)
  result <- jit(value_and_gradient(mlp_loss_nv))(x_nv, y_nv, W1_nv, b1_nv, W2_nv, b2_nv)

  loss_torch <- mlp_loss(x_torch, y_torch, W1_torch, b1_torch, W2_torch, b2_torch, matmul_torch)
  loss_torch$backward()

  # Check forward pass (loss value)
  expect_equal(as_array(result$value), as_array_torch(loss_torch), tolerance = 1e-5)

  # Check reverse pass (gradients for x, y, W1, b1, W2, b2)
  expect_equal(as_array(result$grad[[1L]]), as_array_torch(x_torch$grad), tolerance = 1e-5)
  expect_equal(as_array(result$grad[[3L]]), as_array_torch(W1_torch$grad), tolerance = 1e-5)
  expect_equal(as_array(result$grad[[4L]]), as_array_torch(b1_torch$grad), tolerance = 1e-5)
  expect_equal(as_array(result$grad[[5L]]), as_array_torch(W2_torch$grad), tolerance = 1e-5)
  expect_equal(as_array(result$grad[[6L]]), as_array_torch(b2_torch$grad), tolerance = 1e-5)
})

test_that("polynomial forward and reverse", {
  poly <- function(x) {
    x2 <- x * x
    x3 <- x2 * x
    x4 <- x3 * x
    x5 <- x4 - 2 * x3 + 3 * x2 - 4 * x + 5
    sum(x5)
  }

  x_data <- array(rnorm(8, mean = 1, sd = 0.5), dim = 8L)
  x_nv <- nv_array(x_data, dtype = "f32")
  x_torch <- torch::torch_tensor(x_data, dtype = torch::torch_float32(), requires_grad = TRUE)

  # Forward and reverse pass with value_and_gradient
  result <- jit(value_and_gradient(poly))(x_nv)

  loss_torch <- poly(x_torch)
  loss_torch$backward()

  # Check forward pass
  expect_equal(as_array(result$value), as_array_torch(loss_torch), tolerance = 1e-5)

  # Check reverse pass
  expect_equal(as_array(result$grad[[1L]]), as_array_torch(x_torch$grad), tolerance = 1e-4)
})
