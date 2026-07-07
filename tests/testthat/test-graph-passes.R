describe("inline_scalarish_constants", {
  check_inlining <- function(
    graph_fun,
    args,
    expected_constants_before = NULL,
    expected_constants_after = NULL,
    check_literals = NULL
  ) {
    graph <- do.call(trace_fn, c(list(f = graph_fun), args))

    if (!is.null(expected_constants_before)) {
      expect_length(graph$constants, expected_constants_before)
    }

    new_graph <- inline_scalarish_constants(graph)

    if (!is.null(expected_constants_after)) {
      expect_length(new_graph$constants, expected_constants_after)
    }

    expect_gte(length(new_graph$calls), length(graph$calls))
    expect_equal(length(new_graph$inputs), length(graph$inputs))
    expect_equal(length(new_graph$outputs), length(graph$outputs))
    expect_identical(new_graph$in_tree, graph$in_tree)
    expect_identical(new_graph$out_tree, graph$out_tree)

    if (!is.null(check_literals)) {
      check_literals(new_graph, graph)
    }

    run <- function(graph) {
      out <- stablehlo(graph)
      func <- out[[1L]]
      consts <- out[[2L]]
      const_arrays <- lapply(consts, \(c) {
        arr <- c$aval$data
        if (backend(arr) == "plain") {
          pjrt::pjrt_buffer(as_array(arr), as.character(dtype(arr)), shape = shape(arr))
        } else {
          arr$data
        }
      })
      program <- pjrt::pjrt_program(src = stablehlo::repr(func), format = "mlir")
      exec <- pjrt::pjrt_compile(program)
      inputs_flat <- lapply(flatten(args), \(a) a$data)
      do.call(pjrt::pjrt_execute, c(list(exec), const_arrays, inputs_flat, list(simplify = FALSE)))
    }

    expect_equal(run(graph), run(new_graph))

    return(list(original = graph, processed = new_graph))
  }

  it("converts scalar constants to GraphLiterals", {
    const_scalar <- nv_scalar(5)
    f <- function(x) {
      x + const_scalar
    }

    check_inlining(
      graph_fun = f,
      args = list(list(x = nv_scalar(1))),
      expected_constants_before = 1L,
      expected_constants_after = 0L,
      check_literals = function(new_graph, original_graph) {
        expect_true(is_graph_literal(new_graph$calls[[1L]]$inputs[[2L]]))
        expect_equal(new_graph$calls[[1L]]$inputs[[2L]]$aval$data, const_scalar)
      }
    )
  })

  it("replaces simple scalar", {
    const_scalar <- nv_scalar(5)
    f <- function(x) {
      x + const_scalar
    }

    check_inlining(
      graph_fun = f,
      args = list(list(x = nv_scalar(1))),
    )
  })

  it("replaces scalarish constant", {
    const_scalar <- nv_array(5, shape = c(1, 1, 1))
    f <- function(x) {
      x + const_scalar
    }

    check_inlining(
      graph_fun = f,
      args = list(list(x = nv_scalar(1))),
      expected_constants_before = 1L,
      expected_constants_after = 0L
    )
  })

  it("works if output is scalar constant", {
    f <- function(x) {
      nv_scalar(1)
    }

    result <- check_inlining(
      graph_fun = f,
      args = list(list(x = nv_scalar(0))),
      expected_constants_before = 1L,
      expected_constants_after = 0L
    )
  })
  it("can inline constant inputs to sub-graphs", {
    f <- function() {
      x <- nv_scalar(TRUE)
      nv_if(x, \() nv_scalar(1), \() nv_scalar(2))
    }
    result <- check_inlining(
      graph_fun = f,
      args = list(list()),
      expected_constants_before = 3L,
      expected_constants_after = 0L
    )
  })

  it("replaces all references to converted constants", {
    const_scalar <- nv_scalar(3)
    f <- function(x) {
      y <- x + const_scalar
      y * const_scalar
    }
    graph <- trace_fn(f, list(x = nv_scalar(1)))
    inline_scalarish_constants(graph)

    check_inlining(
      graph_fun = f,
      args = list(list(x = nv_scalar(1))),
      check_literals = function(new_graph, original_graph) {
        # one fill call is added
        expect_true(length(new_graph$calls) == length(original_graph$calls) + 1L)
      }
    )
  })

  it("handles graphs with no scalar constants", {
    f <- function(x, y) {
      x + y
    }

    check_inlining(
      graph_fun = f,
      args = list(list(x = nv_scalar(1), y = nv_scalar(2))),
      expected_constants_before = 0L,
      expected_constants_after = 0L
    )
  })

  it("handles multiple scalar constants", {
    const1 <- nv_scalar(1)
    const2 <- nv_scalar(2)
    const3 <- nv_scalar(3)

    f <- function(x) {
      x + const1 + const2 + const3
    }

    check_inlining(
      graph_fun = f,
      args = list(list(x = nv_scalar(0))),
      expected_constants_before = 3L,
      expected_constants_after = 0L,
      check_literals = function(new_graph, original_graph) {
        expect_true(is_graph_literal(new_graph$calls[[1L]]$inputs[[2L]]))
        expect_equal(new_graph$calls[[1L]]$inputs[[2L]]$aval$data, const1)
      }
    )
  })

  it("preserves dtype of converted literals", {
    const_init <- nv_scalar(1L, dtype = "i8")
    expect_equal(
      jit(\() const_init)(),
      const_init
    )
  })

  it("processes nested subgraphs", {
    # Create a graph with nested nv_if calls
    const_inner_true <- nv_scalar(100)
    const_inner_false <- nv_scalar(200)
    const_outer_true <- nv_scalar(10)
    const_outer_false <- nv_scalar(20)

    f <- function(x, y) {
      nv_if(
        x,
        \() nv_if(y, \() const_inner_true, \() const_inner_false),
        \() nv_if(y, \() const_outer_true, \() const_outer_false)
      )
    }

    g1 <- trace_fn(f, list(x = nv_scalar(TRUE), y = nv_scalar(TRUE)))
    g2 <- inline_scalarish_constants(g1)

    check_inlining(
      graph_fun = f,
      args = list(list(x = nv_scalar(TRUE), y = nv_scalar(TRUE))),
      check_literals = function(new_graph, original_graph) {
        expect_equal(length(new_graph$calls), length(original_graph$calls) + 4L)
      }
    )
  })

  it("inlines identical constants once", {
    x <- nv_scalar(1)
    f <- function() {
      list(x, x)
    }
    check_inlining(
      graph_fun = f,
      args = list(list()),
      expected_constants_before = 1L,
      expected_constants_after = 0L,
      check_literals = function(new_graph, original_graph) {
        expect_equal(length(new_graph$calls), length(original_graph$calls) + 1L)
        expect_identical(new_graph$outputs[[1L]], new_graph$outputs[[2L]])
      }
    )
  })

  it("works in nv_while", {
    f <- function() {
      i <- nv_scalar(0)
      nv_while(list(i = i), \(i) i < nv_scalar(10), \(i) {
        i <- i + nv_scalar(1)
        list(i = i)
      })
    }
    check_inlining(
      graph_fun = f,
      args = list(list()),
      expected_constants_before = 3L,
      expected_constants_after = 0L
    )
  })

  it("works in nv_if", {
    x <- nv_scalar(10)
    y <- nv_scalar(20)
    f <- function(pred) {
      nv_if(pred, \() x, \() y)
    }
    graph <- trace_fn(f, list(pred = nv_scalar(TRUE)))
    check_inlining(
      graph_fun = f,
      args = list(list(pred = nv_scalar(TRUE))),
      expected_constants_before = 2L,
      expected_constants_after = 0L
    )
  })

  it("does not inline non-scalarish constants", {
    y <- nv_array(1:2)
    f <- function() y
    graph <- trace_fn(f, list())
    new_graph <- inline_scalarish_constants(graph)
    expect_length(new_graph$constants, 1L)
  })
})

describe("remove_unused_constants", {
  it("removes unused constants", {
    y <- nv_scalar(5, "f32")
    f <- function(x) {
      x + y
    }
    graph <- transform_gradient(trace_fn(f, list(x = nv_scalar(1, "f32"))), wrt = "x")
    new_graph <- remove_unused_constants(graph)
    # Function should run without error
    expect_true(is_graph(new_graph))
  })

  it("does not remove used constants", {
    y <- nv_scalar(1)
    f <- function(x) {
      x + y
    }
    graph <- trace_fn(f, list(x = nv_scalar(1)))
    expect_length(graph$constants, 1L)
    new_graph <- remove_unused_constants(graph)
    expect_length(new_graph$constants, 1L)
  })
})

describe("resolve_optimization_passes", {
  it("TRUE selects all passes in canonical order", {
    expect_identical(
      resolve_optimization_passes(TRUE),
      names(graph_optimization_passes)
    )
  })

  it("FALSE selects no passes", {
    expect_identical(resolve_optimization_passes(FALSE), character())
  })

  it("a character vector selects a subset, in canonical order", {
    expect_identical(
      resolve_optimization_passes("remove_unused_constants"),
      "remove_unused_constants"
    )
    expect_identical(
      resolve_optimization_passes(c("remove_unused_constants", "inline_scalars")),
      names(graph_optimization_passes)
    )
    expect_identical(resolve_optimization_passes(character()), character())
  })

  it("rejects unknown pass names", {
    expect_error(resolve_optimization_passes("nope"), "Unknown optimization pass")
  })

  it("rejects a non-flag logical", {
    expect_error(resolve_optimization_passes(c(TRUE, FALSE)))
    expect_error(resolve_optimization_passes(NA))
  })
})

describe("optimize_graph", {
  it("FALSE leaves scalar constants in the graph, TRUE inlines them", {
    y <- nv_scalar(1)
    f <- function(x) x + y
    graph <- trace_fn(f, list(x = nv_scalar(1)))
    expect_length(graph$constants, 1L)

    expect_length(optimize_graph(graph, FALSE)$constants, 1L)
    expect_length(optimize_graph(graph, TRUE)$constants, 0L)
    expect_length(
      optimize_graph(graph, "remove_unused_constants")$constants,
      1L
    )
  })
})

describe("trace_fn(optimize = )", {
  it("defaults to no optimization and honours the passes when requested", {
    y <- nv_scalar(1)
    f <- function(x) x + y

    expect_length(trace_fn(f, list(x = nv_scalar(1)))$constants, 1L)
    expect_length(
      trace_fn(f, list(x = nv_scalar(1)), optimize = FALSE)$constants,
      1L
    )
    expect_length(
      trace_fn(f, list(x = nv_scalar(1)), optimize = TRUE)$constants,
      0L
    )
    expect_length(
      trace_fn(
        f,
        list(x = nv_scalar(1)),
        optimize = "remove_unused_constants"
      )$constants,
      1L
    )
  })
})
