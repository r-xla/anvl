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

describe("the functions a lowering builds once", {
  # The draws of different shapes and data types, two of them alike, and a fork.
  draws <- function() {
    list(
      prim_sample_bits(2L, "ui32"),
      prim_sample_bits(2L, "ui32"),
      prim_sample_bits(c(2L, 3L), "ui32"),
      prim_sample_bits(2L, "ui64"),
      prim_fork_random_state()$random_key
    )
  }

  it("builds one function per draw shape and data type, which every such draw calls", {
    out <- stablehlo(trace_fn(draws, list()))
    ids <- vapply(out[[4L]], function(func) sub("^@", "", repr(func$id)), character(1L))
    expect_setequal(ids, c("sample_bits_ui32_2", "sample_bits_ui32_2x3", "sample_bits_ui64_2", "fork_random_state"))
    main <- repr(out[[1L]])
    calls <- function(id) lengths(regmatches(main, gregexpr(paste0("func.call @", id, "\\("), main)))
    expect_identical(calls("sample_bits_ui32_2"), 2L)
    expect_identical(calls("sample_bits_ui32_2x3"), 1L)
    expect_identical(calls("sample_bits_ui64_2"), 1L)
    expect_identical(calls("fork_random_state"), 1L)
  })

  it("gives each draw its own values, those the same calls made eagerly give", {
    read <- function(xs) lapply(xs, function(x) as.numeric(as_array(nv_bitcast_convert(x, dtype = "ui32"))))
    eager <- with_nv_seed(1L, read(draws()))
    jitted <- with_nv_seed(1L, read(jit(draws)()))
    expect_identical(jitted, eager)
    # the two draws alike call the same function at different counters
    expect_false(identical(eager[[1L]], eager[[2L]]))
  })
})

describe("stablehlo() of a graph with side effects", {
  graph <- trace_fn(function(x) x + nv_runif(2L), list(x = nv_aval("f32", 2L)))

  it("takes their values after the inputs and returns none of them the host does not receive", {
    out <- stablehlo(graph)
    func <- out[[1L]]
    # `x`, then the key and the counter of the random state
    expect_length(func$inputs, 3L)
    expect_true(all(vapply(func$inputs, function(input) is.null(input$alias), logical(1L))))
    expect_length(func$outputs, 1L)
  })

  it("never donates a side effect's value", {
    graph <- trace_fn(function(s) s + prim_sample_bits(4L, "ui32"), list(s = nv_aval("ui32", 4L)))
    inputs <- stablehlo(graph, donate = "s")[[1L]]$inputs
    # the donated input `s` is aliased with the output, the random state is not
    expect_identical(inputs[[1L]]$alias, 0L)
    expect_null(inputs[[2L]]$alias)
    expect_null(inputs[[3L]]$alias)
  })

  it("adds no phantom input for the result of a side effect", {
    out <- stablehlo(graph, donate_unaliased_outputs = TRUE, platform = "cpu")
    expect_length(out[[3L]], 1L)
    expect_identical(out[[1L]]$inputs[[4L]]$alias, 0L)
  })
})
