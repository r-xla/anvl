test_that("AnvlPrimitive basics", {
  p <- AnvlPrimitive("abc")
  expect_class(p, "AnvlPrimitive")
  expect_equal(p$name, "abc")
  expect_snapshot(p)
})

test_that("quickr rules are exposed through primitives", {
  expect_true(is.function(prim_add[["quickr"]]))
  expect_null(prim_print[["quickr"]])
})

documented_primitive_ids <- function() {
  primitives_path <- testthat::test_path("..", "..", "R", "primitives.R")
  if (!file.exists(primitives_path)) {
    testthat::skip("R/primitives.R is only available when testing from package source")
  }

  primitive_lines <- readLines(primitives_path)
  sub(
    "^#' @templateVar primitive_id ",
    "",
    grep("^#' @templateVar primitive_id ", primitive_lines, value = TRUE)
  )
}

test_that("documented primitive ids resolve to registered primitives", {
  primitive_ids <- documented_primitive_ids()

  missing <- primitive_ids[vapply(primitive_ids, function(id) is.null(primitive_env[[id]]), logical(1))]
  expect_identical(missing, character())
})

test_that("new_primitive builds a callable that self-registers", {
  on.exit(rm("np_test", envir = primitive_env))

  fn <- new_primitive("np_test", function(x) x + 1)

  expect_class(fn, "JitPrimitive")
  expect_class(fn, "JitFunction")
  expect_identical(primitive_env$np_test, fn)
  expect_identical(attr(fn, "primitive")$name, "np_test")
  expect_identical(formals(fn), formals(function(x) x + 1))
})

test_that("new_primitive respects register = FALSE", {
  fn <- new_primitive("np_unregistered", function(x) x, register = FALSE)
  expect_false(exists("np_unregistered", envir = primitive_env, inherits = FALSE))
})

test_that("JitPrimitive [[ delegates to attached AnvlPrimitive", {
  p <- AnvlPrimitive("jp_test_a")
  f <- function(x) x
  attr(f, "primitive") <- p
  class(f) <- c("JitPrimitive", "function")

  f[["stablehlo"]] <- function(x) "stablehlo-rule"
  expect_identical(p[["stablehlo"]](), "stablehlo-rule")

  expect_identical(f[["stablehlo"]], p[["stablehlo"]])
})

describe("subgraphs", {
  it("extracts subgraphs from higher-order primitives", {
    true_graph <- trace_fn(function() nv_scalar(1), list())
    false_graph <- trace_fn(function() nv_scalar(2), list())
    call <- PrimitiveCall(
      primitive = prim_if,
      inputs = list(GraphValue(aval = nv_aval("bool", integer()))),
      params = list(true = true_graph, false = false_graph),
      outputs = list(GraphValue(aval = nv_aval("f32", integer())))
    )

    subgraphs_list <- subgraphs(call)
    expect_length(subgraphs_list, 2L)
    expect_named(subgraphs_list, c("true", "false"))
    expect_identical(subgraphs_list[["true"]], true_graph)
    expect_identical(subgraphs_list[["false"]], false_graph)
  })
  it("returns empty list for non-higher-order primitives", {
    call <- PrimitiveCall(
      primitive = prim_add,
      inputs = list(GraphValue(aval = nv_aval("f32", integer())), GraphValue(aval = nv_aval("f32", integer()))),
      params = list(),
      outputs = list(GraphValue(aval = nv_aval("f32", integer())))
    )
    expect_length(subgraphs(call), 0L)
  })
})

describe("new_primitive", {
  it("works for a body defined outside the anvl namespace", {
    fn <- function(x) {
      graph_desc_add(self, list(x), infer_fn = function(x) list(x))[[1L]]
    }
    # Not the global env: `load_all()` attaches anvl's internals to the search
    # path, which would hide the internals the body cannot see from a user's
    # package.
    environment(fn) <- list2env(list(graph_desc_add = graph_desc_add), parent = baseenv())
    prim_identity <- new_primitive("identity_ext", fn, register = FALSE)
    expect_s3_class(trace_fn(prim_identity, list(nv_aval("f32", 2L))), "AnvlGraph")
  })
})

describe("the call a primitive's error reports", {
  call_of <- function(expr) {
    err <- tryCatch(expr, error = identity)
    deparse(conditionCall(err))
  }

  it("is the `prim_*()` the caller wrote, whichever helper raised it", {
    # Without this the caller reads `Error in assert_int_param()`,
    # `Error in resolve_axes()` or `Error in (function (init, cond, body)` --
    # the helper a body checks its arguments with, or the anonymous function
    # `jit()` wraps. An inference rule's error is already rewritten this way in
    # `trace_fn()`; this covers everything raised in the wrapper.
    expect_equal(call_of(prim_top_k(nv_array(1:4), 2.5)), "prim_top_k()")
    expect_equal(call_of(prim_rev(nv_array(1:4), 5L)), "prim_rev()")
    expect_equal(call_of(prim_reshape(nv_array(1:4), mean)), "prim_reshape()")
    expect_equal(call_of(prim_chol(nv_array(c(1, 2), dtype = "f32"))), "prim_chol()")
    expect_equal(call_of(prim_fill(NaN, 2L, "i32")), "prim_fill()")
    # Not raised by a checking helper at all: `prim_scan()` asserts with
    # checkmate, and a `NULL` operand dies inside the tracer.
    expect_equal(
      call_of(prim_scan(
        nv_scalar(0L),
        list(nv_array(1:3)),
        function(c, x) list(carry = c + x, out = c),
        steps = NA
      )),
      "prim_scan()"
    )
    expect_equal(
      call_of(prim_concatenate(nv_array(1:4), NULL, axis = 1L)),
      "prim_concatenate()"
    )
    # An inference error, which `trace_fn()` had already attributed.
    expect_equal(
      call_of(prim_reshape(nv_array(1:4), c(3L, 3L))),
      "prim_reshape()"
    )
  })

  it("survives the sub-graphs a higher-order primitive traces", {
    # Every primitive inside `body` names itself and clears the marker again on
    # its way out, so without `trace_fn()` restoring it the check that follows
    # the sub-trace would report no call at all.
    expect_equal(
      call_of(prim_while(
        list(i = nv_scalar(1L)),
        cond = function(i) i < 3L,
        body = function(i) list(nv_convert(i, "f32"))
      )),
      "prim_while()"
    )
    expect_equal(
      call_of(prim_reduce(
        nv_array(1:4),
        nv_scalar(0L),
        1L,
        reducer = function(a, b) nv_convert(a + b, "f32")
      )),
      "prim_reduce()"
    )
  })

  it("is the primitive inside a sub-graph that raised it, not the one tracing it", {
    body_fails <- function(body) {
      call_of(prim_while(list(i = nv_scalar(1L)), cond = function(i) i < 3L, body = body))
    }
    expect_equal(body_fails(function(i) list(prim_reshape(i, c(2L, 2L)))), "prim_reshape()")
    # Raised after the body's primitives have passed, the error keeps the call
    # it was raised in rather than going to the `prim_while()` tracing it.
    call <- body_fails(function(i) {
      i + 1L
      stop("user error")
    })
    expect_false(any(c("prim_while()", "prim_add()") %in% call))
  })

  it("is cleared again, so a later error is not blamed on the last primitive", {
    err <- tryCatch(
      jit(function(a) {
        b <- a + 1L
        stop("user error")
      })(nv_array(1:4)),
      error = identity
    )
    expect_false(identical(deparse(conditionCall(err)), "prim_add()"))
  })
})
