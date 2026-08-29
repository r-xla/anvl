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

test_that("format_param: empty cases collapse to empty string", {
  expect_snapshot({
    format_param(NULL)
    format_param(list())
  })
})

test_that("format_param: atomic scalars", {
  expect_snapshot({
    format_param(1L)
    format_param(1.5)
    format_param(TRUE)
    format_param("abc")
  })
})

test_that("format_param: atomic vectors", {
  expect_snapshot({
    format_param(c(1L, 2L, 3L))
    format_param(c("a", "b"))
    format_param(c(TRUE, FALSE))
  })
})

test_that("format_param: empty atomic vectors show typeof(0)", {
  expect_snapshot({
    format_param(integer())
    format_param(character())
    format_param(logical())
  })
})

test_that("format_param: lists", {
  expect_snapshot({
    format_param(list(1, 2))
    format_param(list(a = 1, b = 2))
  })
})

test_that("format_param: NULL nested in a list is printed as NULL", {
  expect_snapshot({
    format_param(list(NULL, 1))
    format_param(list(a = NULL, b = 1))
  })
})

test_that("format_param: nested lists", {
  expect_snapshot({
    format_param(list(list(x = 1), 2))
    format_param(list(list(c(1, 2, 3))))
    format_param(list(a = list(b = c(2, 3, 4))))
  })
})

test_that("format_param: dtype uses repr", {
  expect_snapshot({
    format_param(as_dtype("f32"))
    format_param(as_dtype("i32"))
  })
})

test_that("format_param: graph is summarized by input/output count", {
  g <- trace_fn(function(x) x + nv_scalar(1, dtype = "f32"), list(x = nv_scalar(0, dtype = "f32")))
  expect_snapshot(format_param(g))
})
