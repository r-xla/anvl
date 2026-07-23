test_that("literals", {
  f <- function(x) {
    x * 1L
  }
  graph <- trace_fn(f, list(x = nv_scalar(1, dtype = "f32")))
  expect_snapshot(graph)

  # higher-dimensional literals
  f <- function() {
    nv_fill(1, shape = c(2, 1))
  }
  graph <- trace_fn(f, list())
  expect_snapshot(graph)
})

test_that("ambiguity is printed via ?", {
  f <- function(x) {
    x * 1
  }
  graph <- trace_fn(f, list(x = nv_scalar(TRUE)))
  expect_snapshot(graph)
})

test_that("constants", {
  y <- nv_scalar(1, dtype = "f32")
  f <- function(x) {
    x + y
  }
  graph <- trace_fn(f, list(x = nv_scalar(2, dtype = "f32")))
  expect_snapshot(graph)
})

test_that("sub-graphs (if)", {
  f <- function(x) {
    nv_if(x, \() nv_scalar(1, dtype = "f32"), \() nv_scalar(2, dtype = "f32"))
  }
  graph <- trace_fn(f, list(x = nv_scalar(TRUE)))
  expect_snapshot(graph)
})

test_that("sub-graphs (while)", {
  f <- function(x) {
    nv_while(list(i = nv_scalar(0, dtype = "f32")), \(i) i < x, \(i) {
      list(i = i + nv_scalar(1, dtype = "f32"))
    })
  }
  graph <- trace_fn(f, list(x = nv_scalar(10, dtype = "f32")))
  expect_snapshot(graph)
})

test_that("params", {
  f <- function(x) {
    nv_reduce_max(x, axes = 1, drop = TRUE)
  }
  graph <- trace_fn(f, list(x = nv_array(1:10)))
  expect_snapshot(graph)
})
