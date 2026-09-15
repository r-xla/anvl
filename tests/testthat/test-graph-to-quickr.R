test_that("graph_to_quickr_function supports list outputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      list(a = x, b = x + x)
    },
    list(x = nv_scalar(1.0, dtype = "f64"))
  )

  expect_quickr_matches_pjrt(graph, 0.5)
})

test_that("graph_to_quickr_function rejects non-graph inputs", {
  expect_error(graph_to_quickr_function(1), "must be a .*AnvlGraph", fixed = FALSE)
})

test_that("graph_to_quickr_r_function rejects non-graph inputs", {
  expect_error(graph_to_quickr_r_function(1), "must be a .*AnvlGraph", fixed = FALSE)
})

test_that("graph_to_quickr_function supports nested inputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      x$a + x$b
    },
    list(
      x = list(
        a = nv_scalar(1.0, dtype = "f64"),
        b = nv_scalar(2.0, dtype = "f64")
      )
    )
  )

  expect_quickr_matches_pjrt(graph, x = list(a = 0.5, b = 1.25))
})

test_that("graph_to_quickr_function handles GraphLiteral inputs (R scalar literals)", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      x + 1L
    },
    list(x = nv_scalar(0.0, dtype = "f64")),
    desc = local_descriptor()
  )

  expect_quickr_matches_pjrt(graph, 2.5)
})

test_that("graph_to_quickr_function produces a stable flat signature", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(out, v1) {
      out + v1
    },
    list(
      out = nv_scalar(1.0, dtype = "f64"),
      v1 = nv_scalar(2.0, dtype = "f64")
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  f_r <- graph_to_quickr_r_function(graph)
  expect_identical(names(formals(f_quick)), c("x1", "x2"))
  expect_identical(names(formals(f_r)), c("x1", "x2"))

  expect_quickr_matches_pjrt(graph, 0.5, 1.25)
})

test_that("graph_to_quickr_function handles zero-length output leaves", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function() {
      list(
        b = nv_fill(2.0, shape = integer(), dtype = "f64"),
        a = nv_fill(1.0, shape = c(0L, 2L), dtype = "f64"),
        c = nv_fill(3.0, shape = integer(), dtype = "f64")
      )
    },
    list()
  )

  out <- expect_quickr_matches_pjrt(graph)
  out_quick <- out$out_quick
  expect_identical(dim(out_quick$a), c(0L, 2L))
})

test_that("graph_to_quickr_function ignores static args when flattening nested inputs", {
  skip_if_no_quickr_or_pjrt()

  # Regression test: when tracing with nested list inputs and an extra static arg,
  # graph$inputs contains only the array leaves but wrapper flattening sees both.
  graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x$a$u + x$b
      } else {
        x$a$v + x$b
      }
    },
    list(
      x = list(
        a = list(
          u = nv_scalar(1.0, dtype = "f64"),
          v = nv_scalar(2.0, dtype = "f64")
        ),
        b = nv_scalar(3.0, dtype = "f64")
      ),
      flag = TRUE
    )
  )

  x <- list(a = list(u = 0.5, v = 1.5), b = 2.0)
  expect_quickr_matches_pjrt(graph, x = x, flag = TRUE)
})

test_that("graph_to_quickr_function accepts flat static args (no nested inputs)", {
  skip_if_no_quickr_or_pjrt()

  fn <- function(x, flag) {
    if (flag) {
      x + 1L
    } else {
      x - 1L
    }
  }

  graph <- trace_fn(
    fn,
    list(
      x = nv_scalar(0.0, dtype = "f64"),
      flag = TRUE
    ),
    desc = local_descriptor()
  )

  f_quick <- graph_to_quickr_function(graph)
  f_r <- graph_to_quickr_r_function(graph)
  expect_identical(names(formals(f_quick)), c("x", "flag"))
  expect_identical(names(formals(f_r)), c("x", "flag"))

  out_true <- expect_quickr_matches_pjrt(graph, x = 0.5, flag = TRUE)
  expect_equal(out_true$out_quick, 1.5)
})

test_that("graph_to_quickr wrappers reject mismatched runtime static args", {
  skip_if_no_quickr()

  flat_graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x + 1L
      } else {
        x - 1L
      }
    },
    list(
      x = nv_scalar(0.0, dtype = "f64"),
      flag = TRUE
    ),
    desc = local_descriptor()
  )

  flat_r <- graph_to_quickr_r_function(flat_graph)
  flat_quick <- graph_to_quickr_function(flat_graph)

  expect_error(flat_r(0.5, FALSE), "Static arguments must match", fixed = FALSE)
  expect_error(flat_quick(0.5, FALSE), "Static arguments must match", fixed = FALSE)

  nested_graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x$a$u + x$b
      } else {
        x$a$v + x$b
      }
    },
    list(
      x = list(
        a = list(
          u = nv_scalar(1.0, dtype = "f64"),
          v = nv_scalar(2.0, dtype = "f64")
        ),
        b = nv_scalar(3.0, dtype = "f64")
      ),
      flag = TRUE
    )
  )

  nested_r <- graph_to_quickr_r_function(nested_graph)
  nested_quick <- graph_to_quickr_function(nested_graph)
  x <- list(a = list(u = 0.5, v = 1.5), b = 2.0)

  expect_error(nested_r(x = x, flag = FALSE), "Static arguments must match", fixed = FALSE)
  expect_error(nested_quick(x = x, flag = FALSE), "Static arguments must match", fixed = FALSE)
})

test_that("graph_to_quickr_function generates placeholder names for unnamed `...` inputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(...) {
      xs <- list(...)
      xs[[1L]] + xs[[2L]]$a + xs[[2L]]$b
    },
    list(
      nv_scalar(0.0, dtype = "f64"),
      list(
        a = nv_scalar(1.0, dtype = "f64"),
        b = nv_scalar(2.0, dtype = "f64")
      )
    )
  )

  # Formals are generated from empty in_tree names; call by position to keep this robust.
  out <- expect_quickr_matches_pjrt(graph, 0.25, list(a = 0.5, b = 1.0))
  expect_equal(out$out_quick, 0.25 + 0.5 + 1.0)
})

test_that("graph_to_quickr_function errors on mismatched flattened input structure", {
  skip_if_no_quickr()

  graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        x$a$u + x$b
      } else {
        x$a$v + x$b
      }
    },
    list(
      x = list(
        a = list(
          u = nv_scalar(1.0, dtype = "f64"),
          v = nv_scalar(2.0, dtype = "f64")
        ),
        b = nv_scalar(3.0, dtype = "f64")
      ),
      flag = TRUE
    )
  )

  f_quick <- graph_to_quickr_function(graph)
  f_r <- graph_to_quickr_r_function(graph)
  x_bad <- list(a = list(u = 0.5), b = 2.0)
  expect_error(
    f_quick(x = x_bad, flag = TRUE),
    "Expected 4 flattened inputs, got 3",
    fixed = FALSE
  )
  expect_error(
    f_r(x = x_bad, flag = TRUE),
    "Expected 4 flattened inputs, got 3",
    fixed = FALSE
  )
})

test_that("graph_to_quickr_function preserves 1D output axes", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      x + 1L
    },
    list(x = nv_array(numeric(3), shape = 3L, dtype = "f64")),
    desc = local_descriptor()
  )

  out <- expect_quickr_matches_pjrt(graph, c(0.5, 1.5, 2.5))
  expect_identical(dim(out$out_quick), c(3L))
})

test_that("graph_to_quickr_function preserves empty 1D output axes without wrapper", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      x
    },
    list(x = nv_array(numeric(0), shape = 0L, dtype = "f64"))
  )

  x <- array(numeric(), dim = 0L)
  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(dim(out_pjrt), 0L)
  expect_identical(dim(f_r(x)), 0L)
  expect_identical(dim(f_quick(x)), 0L)
  expect_identical(f_r(x), out_pjrt)
  expect_identical(as_array(f_quick(x)), out_pjrt)
})

test_that("graph_to_quickr_function preserves rank-1 axes for direct quickr outputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      nv_expm1(x)
    },
    list(x = nv_array(numeric(3), shape = 3L, dtype = "f64"))
  )

  x <- array(c(0.1, 0.2, 0.3), dim = 3L)
  out_pjrt <- quickr_eval_graph_pjrt(graph, x)
  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(dim(out_pjrt), c(3L))
  expect_identical(dim(f_r(x)), c(3L))
  expect_identical(dim(f_quick(x)), c(3L))
  expect_equal(f_r(x), out_pjrt, tolerance = 1e-12)
  expect_equal(as_array(f_quick(x)), out_pjrt, tolerance = 1e-12)
})

test_that("graph_to_quickr_function preserves rank-1 axes for wrapped quickr outputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x, flag) {
      if (flag) {
        nv_log1p(x)
      } else {
        nv_log1p(x)
      }
    },
    list(
      x = nv_array(numeric(3), shape = 3L, dtype = "f64"),
      flag = TRUE
    )
  )

  x <- array(c(0.1, 0.2, 0.3), dim = 3L)
  out_pjrt <- quickr_eval_graph_pjrt(graph, x, TRUE)
  f_r <- graph_to_quickr_r_function(graph)
  f_quick <- graph_to_quickr_function(graph)

  expect_identical(dim(out_pjrt), c(3L))
  expect_identical(dim(f_r(x, TRUE)), c(3L))
  expect_identical(dim(f_quick(x, TRUE)), c(3L))
  expect_equal(f_r(x, TRUE), out_pjrt, tolerance = 1e-12)
  expect_equal(as_array(f_quick(x, TRUE)), out_pjrt, tolerance = 1e-12)
})

test_that("graph_to_quickr_function preserves pred leaves in multiple outputs", {
  skip_if_no_quickr_or_pjrt()

  graph <- trace_fn(
    function(x) {
      list(p = x > 0L, v = x + 1L)
    },
    list(x = nv_scalar(0.0, dtype = "f64")),
    desc = local_descriptor()
  )

  out <- expect_quickr_matches_pjrt(graph, -0.5)
  out <- out$out_quick

  expect_identical(out$p, FALSE)
  expect_equal(out$v, 0.5)
})
