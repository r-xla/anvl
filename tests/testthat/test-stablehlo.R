test_that("stablehlo: basic test", {
  graph <- trace_fn(prim_add, list(lhs = nv_array(1), rhs = nv_array(2)))
  # expect_identical() is buggy ...
  expect_true(identical(graph$outputs, graph$statements[[1]]$outputs))
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

describe("stablehlo() of a graph with side effects", {
  it("takes their values after the inputs and returns none of them the host does not receive", {
    s <- local_steps()
    graph <- trace_fn(function(x) x + s$step(), list(x = nv_aval("i32", 2L)))
    func <- stablehlo(graph)[[1L]]
    # `x`, then the value of the counted side effect
    expect_length(func$inputs, 2L)
    expect_true(all(vapply(func$inputs, function(input) is.null(input$alias), logical(1L))))
    expect_length(func$outputs, 1L)
  })

  it("never donates a side effect's value", {
    s <- local_steps()
    graph <- trace_fn(function(x) x + s$step(), list(x = nv_aval("i32", 4L)))
    inputs <- stablehlo(graph, donate = "x")[[1L]]$inputs
    # the donated input `x` is aliased with the output, the side effect is not
    expect_identical(inputs[[1L]]$alias, 0L)
    expect_null(inputs[[2L]]$alias)
  })

  it("adds no phantom input for the result of a side effect", {
    s <- local_steps()
    graph <- trace_fn(function(x) x + s$step(), list(x = nv_aval("i32", 2L)))
    out <- stablehlo(graph, donate_unaliased_outputs = TRUE, platform = "cpu")
    expect_length(out[[3L]], 1L)
    expect_identical(out[[1L]]$inputs[[3L]]$alias, 0L)
  })
})
