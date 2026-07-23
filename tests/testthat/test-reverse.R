test_that("basic reverse test", {
  f <- function(x, y) {
    prim_add(x, y)
  }
  f_grad <- jit(gradient(f))
  out <- f_grad(nv_scalar(1.0), nv_scalar(2.0))
  expect_equal(out[[1L]], nv_scalar(1.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("simple function works (scalar)", {
  f_grad <- jit(gradient(prim_mul))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(2.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("chain rule works (scalar)", {
  f <- function(x, y) {
    prim_add(prim_mul(x, y), x)
  }

  f_grad <- jit(gradient(f))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(3.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("gradient does not have to depend on input", {
  # This is special, because the input has no influence
  # on the gradient, because the gradient is constant
  f <- function(x, y) {
    prim_add(x, y)
  }

  f_grad <- jit(gradient(f))

  out <- f_grad(
    nv_scalar(1.0),
    nv_scalar(2.0)
  )

  expect_equal(out[[1L]], nv_scalar(1.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("nested inputs", {
  f <- jit(gradient(function(x) {
    prim_mul(x[[1]][[1]], x[[1]][[1]])
  }))
  expect_equal(
    f(list(list(nv_scalar(1)))),
    list(x = list(list(nv_scalar(2))))
  )
})

test_that("no nested outpus", {
  # we expect a scalar output
  # -> check for good error message
})

test_that("constants work (scalar)", {
  f <- jit(gradient(function(x) {
    prim_mul(x, nv_scalar(2))
  }))
  expect_equal(f(nv_scalar(1)), list(x = nv_scalar(2.0)))
})

test_that("broadcasting works", {
  # A scalar operand auto-broadcasts against a non-scalar in elementwise ops.
  # The reverse pass must reduce the broadcasted cotangent back to the scalar's
  # shape.
  f <- function(a, b) sum(nv_mul(a, b))
  g <- jit(gradient(f))
  res <- g(nv_scalar(2), nv_array(c(1, 2, 3), dtype = "f32"))
  # d/da sum(a * b) = sum(b) = 6, reduced back to a scalar
  expect_equal(res$a, nv_scalar(6))
  expect_equal(shape(res$a), integer())
  # d/db sum(a * b) = a broadcast over b's shape
  expect_equal(res$b, nv_array(c(2, 2, 2), dtype = "f32"))
})

test_that("second order gradient (scalar)", {
  # this works only for scalar functions, so this is primarily a stress
  # test for out transformation implementation, not because it's useful in itself.
  f <- function(x) {
    prim_mul(x, x)
  }
  fg2 <- jit(gradient(\(x) gradient(f)(x)[[1L]]))
  expect_equal(
    fg2(nv_scalar(1)),
    list(x = nv_scalar(2.0))
  )
})

test_that("neg works", {
  g <- jit(gradient(prim_negate))
  expect_equal(g(nv_scalar(1))[[1L]], nv_scalar(-1))
})

test_that("names for grad: primitive", {
  g <- jit(gradient(`*`))
  expect_equal(formalArgs2(g), c("e1", "e2"))
  expect_equal(
    g(nv_scalar(2), nv_scalar(1)),
    list(e1 = nv_scalar(1.0), e2 = nv_scalar(2.0))
  )
})

test_that("names for grad: function", {
  f <- function(e1, e2) {
    e1 * e2
  }
  g <- jit(gradient(f))
  expect_equal(formals(g), formals(f))
  result <- g(nv_scalar(2), nv_scalar(1))
  expect_equal(result, list(e1 = nv_scalar(1.0), e2 = nv_scalar(2.0)))
})

# New tests for selective gradients (wrt)

test_that("partial gradient simple", {
  f <- function(lhs, rhs) {
    prim_add(lhs, rhs)
  }
  g <- jit(gradient(f, wrt = "lhs"))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))[[1L]]
  expect_equal(out, nv_scalar(1.0))
})

test_that("gradient accepts integer positions for wrt", {
  f <- function(lhs, rhs) {
    prim_add(lhs, rhs)
  }
  g <- jit(gradient(f, wrt = 1L))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))[[1L]]
  expect_equal(out, nv_scalar(1.0))

  # value_and_gradient also accepts integer positions.
  vg <- jit(value_and_gradient(f, wrt = 2L))
  res <- vg(nv_scalar(3.0), nv_scalar(4.0))
  expect_equal(res$value, nv_scalar(7.0))
  expect_equal(res$grad[[1L]], nv_scalar(1.0))

  # Out-of-range indices are rejected.
  expect_error(gradient(f, wrt = 3L), "out of range")
  expect_error(value_and_gradient(f, wrt = 0L), "out of range")
})

test_that("wrt cannot be '...'", {
  f <- function(x, ...) x
  expect_error(gradient(f, wrt = "..."), "must not contain")
  expect_error(value_and_gradient(f, wrt = 2L), "must not contain")
})

#test_that("partial gradient: y = a * (x * b) wrt x", {
#  f <- function(a, x, b) {
#    a * (x * b)
#  }
#  g <- gradient(f, wrt = "x")
#  a <- nv_scalar(2.0)
#  x <- nv_scalar(3.0)
#  b <- nv_scalar(5.0)
#  out <- g(a, x, b)
#  expect_null(out[[1L]])
#  expect_equal(out[[2L]], nv_scalar(10.0))
#  expect_null(out[[3L]])
#})
#

#test_that("reverse", {
#  fbwd <- jit(pullback(nv_add, lhs = nv_scalar(1.0), rhs = nv_scalar(2.0), wrt = "lhs"))
#  expect_equal(fbwd(nv_scalar(10.0)), list(lhs = nv_scalar(10.0)))
#})
#
#test_that("pullback: non-scalar", {
#  fbwd <- jit(pullback(nv_mul, lhs = nv_array(1:10), rhs = nv_array(2:11), wrt = "lhs"))
#  x <- nv_array(1:10)
#  expect_equal(fbwd(x), list(lhs = jit(nv_mul)(x, nv_array(2:11))))
#})

test_that("gradients are present even if they don't influence the output", {
  g <- jit(gradient(function(x, y) x, wrt = "y"))
  expect_equal(
    g(nv_scalar(1), nv_scalar(1)),
    list(y = nv_scalar(0))
  )

  g2 <- jit(gradient(function(x, y) {
    z <- nv_mul(x, x)
    return(y)
  }))
  expect_equal(
    g2(nv_scalar(1), nv_scalar(1)),
    list(x = nv_scalar(0.0), y = nv_scalar(1.0))
  )
})

test_that("wrt non-existent argument", {
  f <- function(x) {
    nv_pow(x, nv_scalar(1))
  }
  expect_error(
    jit(gradient(f, wrt = "y"))(nv_scalar(2)),
    "wrt must be"
  )
})

test_that("gradient: simple example", {
  f <- function(x, y) {
    prim_mul(x, y)
  }
  g <- jit(gradient(f))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))
  expect_equal(out[[1L]], nv_scalar(2.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("gradient: does not depend on input", {
  f <- function(x, y) {
    prim_add(x, y)
  }
  g <- jit(gradient(f))
  out <- g(nv_scalar(1.0), nv_scalar(2.0))
  expect_equal(out[[1L]], nv_scalar(1.0))
  expect_equal(out[[2L]], nv_scalar(1.0))
})

test_that("wrt for non-array input: gradient", {
  expect_snapshot(error = TRUE, {
    g <- gradient(nv_round, wrt = "method")
    g(nv_scalar(1), method = "nearest_even")
  })
})

test_that("wrt for non-array input: value_and_gradient", {
  expect_snapshot(error = TRUE, {
    g <- value_and_gradient(nv_round, wrt = "method")
    g(nv_scalar(1), method = "nearest_even")
  })
})

test_that("wrt for nested non-array input: gradient", {
  f <- function(x) {
    prim_mul(x[[1]], x[[2]])
  }
  expect_snapshot(error = TRUE, {
    g <- gradient(f, wrt = "x")
    g(x = list(nv_scalar(1), 2L))
  })
})

test_that("wrt for nested non-array input: value_and_gradient", {
  f <- function(x) {
    prim_mul(x[[1]], x[[2]])
  }
  expect_snapshot(error = TRUE, {
    g <- value_and_gradient(f, wrt = "x")
    g(x = list(nv_scalar(1), 2L))
  })
})

test_that("can only compute gradient w.r.t. float arrays", {
  expect_snapshot(error = TRUE, {
    gradient(nv_floor, wrt = "x")(nv_scalar(1L))
  })
})

test_that("wrt arg passed as plain R literal errors clearly", {
  expect_snapshot(error = TRUE, {
    jit(function() gradient(nv_log, wrt = "x")(1))()
  })
  expect_snapshot(error = TRUE, {
    jit(function() gradient(function(x, y) prim_add(x, y))(1, 2))()
  })
})

test_that("can differentiate through integer/bool functions", {
  f <- function(x) {
    x1 <- nv_convert(x, "i32")
    x2 <- prim_popcnt(x1)
    x3 <- nv_convert(x2, "f32")
    mean(x3)
  }
  g <- jit(gradient(f))
  expect_equal(
    g(nv_array(c(1, 2))),
    list(x = nv_array(c(0, 0)))
  )
})

test_that("gradient with static (non-array) argument", {
  f <- function(x, y) {
    if (x) y * y else y * 7
  }
  g <- jit(gradient(f, wrt = "y"), static = "x")

  # x=TRUE -> y*y -> dy/dy = 2*y = 6
  out_true <- g(TRUE, nv_scalar(3.0))
  expect_equal(out_true[[1L]], nv_scalar(6.0))

  # x=FALSE -> y*7 -> dy/dy = 7
  out_false <- g(FALSE, nv_scalar(3.0))
  expect_equal(out_false[[1L]], nv_scalar(7.0))
})

test_that("value_and_gradient with static (non-array) argument", {
  f <- function(x, y) {
    if (x) y * y else y * 7
  }
  vg <- jit(value_and_gradient(f, wrt = "y"), static = "x")

  # x=TRUE -> y*y = 9, dy/dy = 6
  result_true <- vg(TRUE, nv_scalar(3.0))
  expect_equal(result_true$value, nv_scalar(9.0))
  expect_equal(result_true$grad[[1L]], nv_scalar(6.0))

  # x=FALSE -> y*7 = 21, dy/dy = 7
  result_false <- vg(FALSE, nv_scalar(3.0))
  expect_equal(result_false$value, nv_scalar(21.0))
  expect_equal(result_false$grad[[1L]], nv_scalar(7.0))
})

test_that("Can propagate ambiguous float32 through integer/bool functions", {
  f <- function(x) {
    x1 <- nv_convert(x, "i32")
    x2 <- nv_convert(x1, "bool")
    x3 <- prim_not(x1)
    x4 <- nv_convert(x3, "f32")
    mean(x4)
  }
  grad <- jit(gradient(f))
  out <- grad(nv_scalar(1))
  # the path runs entirely through integer/bool conversions, so no gradient
  # flows back to the float input.
  expect_equal(out$x, nv_scalar(0))
})

test_that("trace_fn matches args with formals", {
  graph1 <- trace_fn(prim_add, list(nv_aval("f32", c()), nv_aval("f32", c())))
  graph2 <- trace_fn(prim_add, list(lhs = nv_aval("f32", c()), rhs = nv_aval("f32", c())))
  expect_true(pjrt::tree_equal(graph1$in_tree, graph2$in_tree))
  expect_equal(pjrt::tree_child_names(graph1$in_tree), c("lhs", "rhs"))
})

test_that("gradient works through graph with primitives that have no reverse rule", {
  # `prim_fill` has no reverse rule; the call should be skipped during the
  # backward pass because its inputs (all static params) don't require grad.
  f <- function(x) {
    y <- prim_fill(2, dtype = "f64", shape = c(3L))
    sum(prim_mul(x, y))
  }
  g <- jit(gradient(f))
  out <- g(nv_array(c(1, 2, 3), dtype = "f64"))
  expect_equal(out[[1L]], nv_array(c(2, 2, 2), dtype = "f64"))
})

test_that("rule_reverse(forward = ...) emits multiple primitives and captures an intermediate", {
  infer <- function(x) list(x)
  my_exp <- new_primitive(
    "my_exp_test",
    function(x) graph_desc_add(self, list(x = x), infer_fn = infer)[[1L]],
    register = FALSE
  )

  my_exp[["reverse"]] <- rule_reverse(forward = function(inputs, params) {
    x <- inputs[[1L]]
    y <- prim_exp(x)
    neg_y <- prim_negate(y)
    list(
      outputs = list(y),
      backward = function(inputs, outputs, grads, params, required) {
        list(if (required[[1L]]) prim_negate(prim_mul(grads[[1L]], neg_y)))
      }
    )
  })

  f <- function(x) sum(my_exp(x))
  g <- jit(gradient(f))
  out <- g(nv_array(c(0, 1), dtype = "f64"))
  expect_equal(out[[1L]], nv_array(c(exp(0), exp(1)), dtype = "f64"))
})
