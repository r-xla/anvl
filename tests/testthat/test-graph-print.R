test_that("literals", {
  local_registered_default_dtypes()
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
  local_registered_default_dtypes()
  f <- function(x) {
    nv_reduce_max(x, axes = 1, drop = TRUE)
  }
  graph <- trace_fn(f, list(x = nv_array(1:10)))
  expect_snapshot(graph)
})

test_that("format_param_parts: a call with no parameters has no parts", {
  expect_snapshot({
    format_param_parts(NULL)
    format_param_parts(list())
  })
})

test_that("format_param_value: atomic scalars", {
  expect_snapshot({
    format_param_value(1L)
    format_param_value(1.5)
    format_param_value(TRUE)
    format_param_value("abc")
  })
})

test_that("format_param_value: atomic vectors", {
  expect_snapshot({
    format_param_value(c(1L, 2L, 3L))
    format_param_value(c("a", "b"))
    format_param_value(c(TRUE, FALSE))
  })
})

test_that("format_param_value: empty atomic vectors show typeof(0)", {
  expect_snapshot({
    format_param_value(integer())
    format_param_value(character())
    format_param_value(logical())
  })
})

test_that("format_param_value: lists", {
  expect_snapshot({
    format_param_value(list(1, 2))
    format_param_value(list(a = 1, b = 2))
  })
})

test_that("format_param_value: NULL nested in a list is printed as NULL", {
  expect_snapshot({
    format_param_value(list(NULL, 1))
    format_param_value(list(a = NULL, b = 1))
  })
})

test_that("format_param_value: nested lists", {
  expect_snapshot({
    format_param_value(list(list(x = 1), 2))
    format_param_value(list(list(c(1, 2, 3))))
    format_param_value(list(a = list(b = c(2, 3, 4))))
  })
})

test_that("format_param_value: dtype prints under its anvl name", {
  expect_snapshot({
    format_param_value(as_dtype("f32"))
    format_param_value(as_dtype("i32"))
  })
})

test_that("format_param_value: graph is summarized by input/output count", {
  g <- trace_fn(function(x) x + nv_scalar(1, dtype = "f32"), list(x = nv_scalar(0, dtype = "f32")))
  expect_snapshot(format_param_value(g))
})

test_that("an input the caller supplies as bare R data names its R type", {
  f <- function(x, y) x + y
  graph <- trace_fn(
    f,
    list(x = nv_scalar(1, dtype = "f64"), y = nv_aval("double", integer()))
  )
  expect_snapshot(graph)

  graph <- trace_fn(f, list(x = nv_aval("f32", integer()), y = nv_aval("integer", 2L)))
  expect_snapshot(graph)
})

test_that("a data type prints under its anvl name, not its MLIR spelling", {
  f <- function(x) x
  graph <- trace_fn(f, list(x = nv_aval("bool", 2L)))
  expect_match(format(graph), "bool[2]", fixed = TRUE)
  expect_no_match(format(graph), "i1", fixed = TRUE)
})

test_that("format_param_value: named atomic vectors keep their names", {
  expect_snapshot({
    format_param_value(c(a = 1, b = 2))
    format_param_value(c(a = 1L))
  })
})

test_that("format_param_value: character values are escaped", {
  expect_snapshot({
    format_param_value("a\"b")
    format_param_value("a\\b")
  })
})

test_that("format_param_value: an array parameter prints as an array", {
  expect_snapshot({
    format_param_value(nv_scalar(1, dtype = "f32"))
    format_param_value(nv_array(c(1, 2, 3), dtype = "f32"))
  })
})

test_that("a call whose parameters do not fit the width wraps them", {
  local_registered_default_dtypes()
  f <- function(x) nv_array(c(1, 2, 3))[x]
  graph <- trace_fn(f, list(x = nv_scalar(1L, dtype = "i64")))
  # `expect_snapshot()` fixes the width at 80, which `gather` overruns.
  expect_snapshot(graph)
  # Given room, the same call stays on one line.
  wide <- withr::with_options(list(width = 300L), format(graph))
  expect_match(wide, "gather [slice_sizes = 1,", fixed = TRUE)
  expect_no_match(wide, "gather [\n", fixed = TRUE)
})
