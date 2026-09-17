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

test_that("format_param_value: a one-element array parameter prints its value", {
  # `inline_scalarish_constants()` hands `prim_fill()` the constant itself as
  # its `value`, and a one-element array can have any all-ones shape.
  expect_snapshot({
    format_param_value(nv_scalar(1, dtype = "f32"))
    format_param_value(nv_array(1, shape = c(1, 1), dtype = "f32"))
    format_param_value(nv_array(c(1, 2, 3), dtype = "f32"))
  })
})

test_that("a folded constant prints its value in the `fill` it becomes", {
  local_registered_default_dtypes()
  y <- nv_scalar(2, dtype = "f32")
  graph <- inline_scalarish_constants(
    trace_fn(function(x) x * y, list(x = nv_scalar(1, dtype = "f32")))
  )
  fill <- graph$calls[[length(graph$calls)]]
  expect_true(is_anvl_array(fill$params$value))
  expect_snapshot(graph)
})

test_that("a literal a call produces is referred to by its node id", {
  local_registered_default_dtypes()
  y <- nv_scalar(2, dtype = "f32")
  graph <- inline_scalarish_constants(
    trace_fn(function(x) x * y, list(x = nv_scalar(1, dtype = "f32")))
  )
  # The `fill` the pass adds writes to a literal. It is still an output of a
  # call, so it gets an id, and the `mul` consuming it points at that id --
  # otherwise both ends print the value and the edge between them disappears.
  out <- format(graph)
  expect_match(out, "%2: f32[] = fill", fixed = TRUE)
  expect_match(out, "mul(%x1, %2)", fixed = TRUE)
  expect_no_match(out, "2:f32[]: ", fixed = TRUE)
})

test_that("a literal no call produces still prints its value", {
  local_registered_default_dtypes()
  graph <- trace_fn(function(x) x + 1, list(x = nv_scalar(1, dtype = "f32")))
  expect_match(format(graph), "add(%x1, 1:f32[])", fixed = TRUE)
})

test_that("a node no call produces and no id prints as `???`", {
  graph <- trace_fn(function(x) x + x, list(x = nv_aval("f32", 2L)))
  graph$outputs <- c(graph$outputs, list(GraphValue(AbstractArray(as_dtype("f32"), 5L))))
  expect_match(format(graph), "???: f32[5]", fixed = TRUE)
})

test_that("format_param_value: a partially named vector names only what has a name", {
  expect_snapshot({
    format_param_value(c(a = 1, 2))
    format_param_value(stats::setNames(c(1, 2), c("", "b")))
    format_param_value(stats::setNames(1, ""))
  })
})

test_that("format_param_value: a partially named list names only what has a name", {
  expect_snapshot({
    format_param_value(list(a = 1, 2))
    format_param_parts(list(a = 1, 2))
  })
})

test_that("format_param_value: each element of a vector is formatted on its own", {
  # `format()` on a whole vector picks one format for all of it, which would
  # print these as `c(1.0, 2.5)`, `c(1e+00, 1e+10)` and `c(1e-01, 1e-20)`.
  expect_snapshot({
    format_param_value(c(1, 2.5))
    format_param_value(c(1, 1e10))
    format_param_value(c(0.1, 1e-20))
  })
})

test_that("a call whose parameters do not fit the width fills them over further lines", {
  local_registered_default_dtypes()
  f <- function(x) nv_array(c(1, 2, 3))[x]
  graph <- trace_fn(f, list(x = nv_scalar(1L, dtype = "i64")))
  # `gather` carries nine parameters and overruns even the `CALL_WIDTH_MIN`
  # floor the layout never goes below.
  expect_snapshot(graph)
  # Every line takes as many parameters as fit, so the list opens on the call's
  # own line and closes on the line the inputs follow.
  out <- format(graph)
  expect_true(all(nchar(strsplit(out, "\n")[[1L]]) <= 120L))
  expect_match(out, "gather [slice_sizes = 1, offset_axes = integer(0),", fixed = TRUE)
  expect_match(out, "unique_indices = TRUE] (%c1, %2)", fixed = TRUE)
  # Given room, the same call stays on one line.
  wide <- withr::with_options(list(width = 300L), format(graph))
  expect_match(wide, "gather [slice_sizes = 1,", fixed = TRUE)
  expect_no_match(wide, "\n      x_batching_axes", fixed = TRUE)
})

test_that("a narrow console does not wrap a short parameter list", {
  local_registered_default_dtypes()
  graph <- trace_fn(function(x) x + 1, list(x = nv_aval("f32", 3L)))
  # A scalar broadcast is 85 characters, so at the console's own width it would
  # cost four lines in every graph that adds a number to an array.
  narrow <- withr::with_options(list(width = 40L), format(graph))
  expect_match(narrow, "broadcast_in_axes [shape = 3, broadcast_axes = integer(0)]", fixed = TRUE)
  expect_no_match(narrow, "broadcast_in_axes [\n", fixed = TRUE)
})

test_that("a call that overruns on its inputs alone is not wrapped", {
  local_registered_default_dtypes()
  args <- stats::setNames(
    replicate(12L, nv_aval("f32", 2L), simplify = FALSE),
    paste0("a", seq_len(12L))
  )
  graph <- trace_fn(function(...) nv_concatenate(..., axis = 1), args)
  # Wrapping moves the parameters off the line and leaves the dozen inputs
  # where they were, so it would buy three lines and no room.
  out <- withr::with_options(list(width = 40L), format(graph))
  expect_match(out, "concatenate [axis = 1] (%x1,", fixed = TRUE)
  expect_no_match(out, "concatenate [\n", fixed = TRUE)
})

test_that("fill_parts: an empty parameter list is just the head and the tail", {
  expect_identical(fill_parts(character(), "op [", "] (%x1)", "  ", 120L), "op [] (%x1)")
})

test_that("fill_parts: a line always takes at least one parameter", {
  # Nothing can split a single parameter wider than the budget, so it overruns
  # on a line of its own rather than looping or dropping out.
  wide <- strrep("x", 60L)
  expect_identical(
    fill_parts(c(wide, "a = 1"), "op [", "] (%x1)", "  ", 20L),
    c(sprintf("op [%s,", wide), "  a = 1] (%x1)")
  )
})

test_that("fill_parts: the budget counts console columns, not characters", {
  # A CJK character is one character but two columns, so counting characters
  # would leave the line half again as wide as the budget allows.
  wide <- sprintf('lab = "%s"', strrep("中", 20L))
  expect_length(fill_parts(c(wide, "a = 1"), "op [", "] (%x1)", "  ", 50L), 2L)
})

test_that("format_param_value: an NA name is not a name", {
  expect_identical(format_param_value(stats::setNames(c(1, 2), c("a", NA))), "c(a = 1, 2)")
})

test_that("a list parameter is spelled list(), so brackets stay the call's own", {
  # `[...]` delimits the parameter group and an aval's shape, so a list in
  # brackets would read as a vector.
  expect_identical(format_param_value(list(a = list(1))), "list(a = list(1))")
  expect_identical(format_param_value(list()), "list()")
  graph <- trace_fn(
    function(x, y) nv_matmul(x, y),
    list(x = nv_aval("f32", c(2L, 3L)), y = nv_aval("f32", c(3L, 4L)))
  )
  # `dot_general` takes one contracting axis per operand, not the pair c(2, 1).
  expect_match(format(graph), "[contracting_axes = list(2, 1),", fixed = TRUE)
})
