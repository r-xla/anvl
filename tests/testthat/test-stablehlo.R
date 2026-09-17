test_that("stablehlo: basic test", {
  graph <- trace_fn(prim_add, list(lhs = nv_array(1), rhs = nv_array(2)))
  # expect_identical() is buggy ...
  expect_true(identical(graph$outputs, graph$calls[[1]]$outputs))
  out <- stablehlo(graph)
})

test_that("stablehlo: a constant", {
  x <- nv_scalar(1)
  f <- function(y) {
    x + y
  }
  graph <- trace_fn(f, list(y = nv_scalar(2)))
  out <- stablehlo(graph)
  func <- out[[1L]]
  const <- out[[2L]][[1L]]
  expect_true(is_graph_value(const))
  expect_identical(const$aval$data, x)
})

test_that("donate: simple example", {
  graph <- trace_fn(identity, list(x = nv_array(3:4, dtype = "i32")))
  out <- stablehlo(graph, donate = "x")
  expect_equal(out[[1]]$inputs[[1]]$alias, 0L)
})

test_that("donate: multiple inputs, only some donated", {
  f <- function(x, y) x + y
  graph <- trace_fn(
    f,
    list(
      x = nv_matrix(1:4, nrow = 2),
      y = nv_matrix(5:8, nrow = 2)
    )
  )
  out <- stablehlo(graph, donate = "x")
  expect_true(
    out[[1]]$inputs[[1]]$alias == 0L || out[[1]]$inputs[[2]]$alias == 0L
  )
})

test_that("donate: nested list inputs", {
  f <- function(x) list(x[[1]], x[[2]])
  graph <- trace_fn(f, list(x = list(nv_array(1), nv_array(2))))
  out <- stablehlo(graph, donate = "x")
  expect_permutation(
    sapply(1:2, \(i) out[[1]]$inputs[[i]]$alias),
    c(0L, 1L)
  )
})

test_that("hand-built regions declare their block arguments with auto value ids", {
  # A region's block arguments live in the SSA namespace of the function the
  # region is nested in, so a fixed name (`%i`) is redefined as soon as the
  # region ends up inside another one using the same name. Auto value ids are
  # numbered across the whole program and cannot collide. `nv_lu()` reaches
  # the hand-built `while` region in `pivots_to_permutation()`.
  graph <- trace_fn(
    function(a) nv_lu(a)$L,
    list(a = nv_aval("f32", shape = c(2L, 2L)))
  )
  src <- stablehlo::repr(stablehlo(graph)[[1L]])

  block_args <- unlist(regmatches(src, gregexpr("\\^bb0\\([^)]*\\)", src)))
  expect_gt(length(block_args), 0L)
  named <- unlist(regmatches(
    block_args,
    gregexpr("%[A-Za-z_][A-Za-z0-9_]*", block_args)
  ))
  expect_equal(named, character())
})
