describe("vectorize", {
  x <- nv_array(matrix(1:6, nrow = 3L), dtype = "f32")

  it("maps over the arguments in in_axes and passes the others to every application", {
    f <- function(x, y) sum(x * y)
    out <- jit(vectorize(f, in_axes = list(x = 1L)))(x, nv_array(c(1, 2), dtype = "f32"))
    expect_equal(as_r(out), array(c(9, 12, 15), dim = 3L))
  })

  it("maps over all arguments by default", {
    f <- function(x, y) sum(x * y)
    out <- jit(vectorize(f))(x, x)
    expect_equal(as_r(out), array(rowSums(as_r(x)^2), dim = 3L))
  })

  it("maps over each argument along its own axis", {
    f <- function(x, y) sum(x * y)
    out <- jit(vectorize(f, in_axes = list(x = 1L, y = 2L)))(x, t(as_r(x)))
    expect_equal(as_r(out), array(rowSums(as_r(x)^2), dim = 3L))
  })

  it("takes in_axes as a named vector and passes arguments named with NULL on", {
    f <- function(x, y) sum(x * y)
    y <- nv_array(c(1, 2), dtype = "f32")
    expect_equal(as_r(jit(vectorize(f, in_axes = c(x = 1L)))(x, y)), array(c(9, 12, 15), dim = 3L))
    expect_equal(as_r(jit(vectorize(f, in_axes = list(x = 1L, y = NULL)))(x, y)), array(c(9, 12, 15), dim = 3L))
  })

  it("stacks each output along the axis out_axes gives it", {
    f <- function(x) list(a = x * 2L, b = list(c = x, d = sum(x)))
    out <- jit(vectorize(f, out_axes = list(a = 2L, b = list(c = 1L, d = 1L))))(x)
    expect_equal(as_r(out$a), t(as_r(x)) * 2)
    expect_equal(as_r(out$b$c), as_r(x))
    expect_equal(as_r(out$b$d), array(rowSums(as_r(x)), dim = 3L))
    g <- function(x) list(a = x * 2L, b = list(c = x, d = x + 1L))
    out <- jit(vectorize(g, out_axes = list(a = 1L, b = 2L)))(x)
    expect_equal(as_r(out$b$c), t(as_r(x)))
    expect_equal(as_r(out$b$d), t(as_r(x)) + 1)
  })

  it("returns an output that out_axes gives NULL as it is", {
    w <- nv_array(c(1, 2), dtype = "f32")
    f <- function(x, w) list(x * w, w * 2L)
    out <- jit(vectorize(f, in_axes = list(x = 1L), out_axes = list(1L, NULL)))(x, w)
    expect_equal(as_r(out[[2L]]), array(c(2, 4), dim = 2L))
  })

  it("rejects out_axes NULL for an output that depends on the arguments mapped over", {
    expect_error(
      jit(vectorize(function(x) list(a = x), out_axes = list(a = NULL)))(x),
      "Output `a` does"
    )
  })

  it("rejects out_axes that do not follow the structure of the output", {
    expect_error(
      jit(vectorize(function(x) list(a = x), out_axes = list(b = 1L)))(x),
      "does not match the output at the top"
    )
    expect_error(
      jit(vectorize(function(x) list(a = list(b = x)), out_axes = list(a = list(1L, 2L))))(x),
      "at `a`"
    )
  })

  it("matches out_axes to a partly named output by position", {
    out <- jit(vectorize(function(x) list(a = x, sum(x)), out_axes = list(a = 2L, 1L)))(x)
    expect_equal(as_r(out$a), t(as_r(x)))
  })

  it("names an argument passed through dots by its position", {
    f <- function(...) ..1 + sum(..2)
    expect_error(jit(vectorize(f))(x, nv_array(1:2, dtype = "f32")), "`argument 2` has size 2")
  })

  it("checks the attribute batched a rule returns", {
    prim_bad <- new_primitive(
      "bad",
      function(x) graph_desc_add(self, list(x = x), infer_fn = infer_numeric_uni)[[1L]],
      register = FALSE
    )
    prim_bad[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
      structure(list(inputs[[1L]]), batched = c(TRUE, FALSE))
    })
    expect_error(jit(vectorize(prim_bad))(x), "invalid attribute")
  })

  it("matches out_axes to named outputs by name", {
    f <- function(x) list(a = x, b = x * 2L)
    out <- jit(vectorize(f, out_axes = list(b = 1L, a = 2L)))(x)
    expect_equal(as_r(out$a), t(as_r(x)))
    expect_equal(as_r(out$b), as_r(x) * 2)
  })

  it("maps over the arrays passed through dots", {
    f <- function(...) {
      args <- list(...)
      sum(args[[1L]] * args[[2L]])
    }
    out <- jit(vectorize(f))(x, x)
    expect_equal(as_r(out), array(rowSums(as_r(x)^2), dim = 3L))
  })

  it("maps over another axis and stacks the outputs along it", {
    f <- function(x) x * 2L
    out <- jit(vectorize(f, in_axes = 2L, out_axes = 2L))(x)
    expect_equal(as_r(out), as_r(x) * 2)

    g <- function(x) cumsum(x)
    out <- jit(vectorize(g, in_axes = 2L, out_axes = 2L))(x)
    expect_equal(as_r(out), apply(as_r(x), 2L, cumsum))
  })

  it("maps over every leaf of a nested argument", {
    f <- function(p) p$a + p$b$c
    out <- jit(vectorize(f))(list(a = x, b = list(c = x)))
    expect_equal(as_r(out), as_r(x) * 2)
  })

  it("returns the structure of f's output", {
    f <- function(x) list(total = sum(x), twice = x * 2L)
    out <- jit(vectorize(f))(x)
    expect_equal(names(out), c("total", "twice"))
    expect_equal(as_r(out$total), array(rowSums(as_r(x)), dim = 3L))
    expect_equal(as_r(out$twice), as_r(x) * 2)
  })

  it("broadcasts an output that does not depend on the arguments mapped over", {
    out <- jit(vectorize(function(x) nv_scalar(1, dtype = "f32")))(x)
    expect_equal(as_r(out), array(1, dim = 3L))
  })

  it("passes static arguments through", {
    f <- function(x, p) sum(x^p)
    out <- jit(vectorize(f, in_axes = list(x = 1L)), static = "p")(x, p = 2L)
    expect_equal(as_r(out), array(rowSums(as_r(x)^2), dim = 3L))
  })

  it("passes static arguments through when mapping over all arguments", {
    f <- function(x, p) sum(x^p)
    out <- jit(vectorize(f), static = "p")(x, p = 2L)
    expect_equal(as_r(out), array(rowSums(as_r(x)^2), dim = 3L))
  })

  it("maps over R data", {
    f <- function(x) sum(x)
    out <- jit(vectorize(f))(matrix(c(1, 2, 3, 4), nrow = 2L))
    expect_equal(as_r(out), array(c(4, 6), dim = 2L))
  })

  it("composes with gradient() into per-instance gradients", {
    loss <- function(w, x) sum((x * w)^2L)
    w <- nv_array(c(1, 2), dtype = "f32")
    out <- jit(vectorize(gradient(loss, wrt = "w"), in_axes = list(x = 1L)))(w, x)
    xr <- as_r(x)
    expect_equal(as_r(out$w), 2 * xr^2 * rep(c(1, 2), each = 3L))
  })

  it("is differentiable", {
    f <- function(w, x) sum(vectorize(function(x) sum(x * w), in_axes = list(x = 1L))(x)^2L)
    w <- nv_array(c(1, 2), dtype = "f32")
    out <- jit(gradient(f, wrt = "w"))(w, x)
    xr <- as_r(x)
    expect_equal(as_r(out$w), array(2 * colSums(drop(xr %*% c(1, 2)) * xr), dim = 2L))
  })

  it("nests", {
    f <- function(a, b) a * b
    f_ab <- vectorize(vectorize(f, in_axes = list(b = 1L)), in_axes = list(a = 1L))
    out <- jit(f_ab)(nv_array(c(1, 2, 3), dtype = "f32"), nv_array(c(10, 20), dtype = "f32"))
    expect_equal(as_r(out), outer(c(1, 2, 3), c(10, 20)))
  })

  it("must be called inside jit", {
    expect_error(vectorize(function(x) x)(x), "inside a")
  })

  it("rejects mapping over an argument that was not passed", {
    expect_error(
      jit(vectorize(function(a, w = a) a * w, in_axes = list(w = 1L)))(x),
      "Cannot map over `w`: it was not passed"
    )
  })

  it("needs an argument to map over", {
    expect_error(
      jit(vectorize(function(x, s) s), static = c("x", "s"))(x = "a", s = 2L),
      "needs at least one argument to map over"
    )
  })

  it("names the argument that cannot be mapped over", {
    f <- function(x, s) x * s
    expect_error(jit(vectorize(f))(x, 2), "`s` has shape ()")
  })

  it("rejects in_axes that are not a single axis or a named list", {
    expect_error(vectorize(function(x) x, in_axes = c(1L, 2L)), "in_axes")
    expect_error(vectorize(function(x) x, in_axes = list(1L)), "named by the arguments")
    expect_error(vectorize(function(x) x, in_axes = list(x = 0L)), "in_axes\\$x")
    expect_error(vectorize(function(x) x, in_axes = list(x = NULL)), "at least one argument")
    expect_error(vectorize(function(x) x, in_axes = list()), "at least one argument")
  })

  it("rejects out_axes that are a vector", {
    expect_error(vectorize(function(x) x, out_axes = c(1L, 2L)), "not a vector")
    expect_error(vectorize(function(x) x, out_axes = c(a = 1L)), "not a vector")
  })

  it("rejects arguments that are not formal arguments of f", {
    expect_error(vectorize(function(x) x, in_axes = list(y = 1L)), "subset of the formal arguments")
  })

  it("rejects arguments of different sizes along the axis", {
    f <- function(x, y) x + y
    expect_error(
      jit(vectorize(f))(x, nv_array(1:4, shape = c(2L, 2L), dtype = "f32")),
      "same size along its batch axis"
    )
  })

  it("rejects an output without room for the batch axis at axis", {
    expect_error(
      jit(vectorize(function(x) sum(x), out_axes = 2L))(x),
      "room for its batch axis"
    )
  })

  it("keeps the device of an array f closes over", {
    local_unset_default_device()
    cst <- nv_array(c(10, 20), dtype = "f32", device = "cpu:1")
    out <- jit(vectorize(function(x) x + cst))(as_r(x)[, 1L:2L])
    expect_equal(device(out), nv_device("cpu:1"))
  })

  it("calls a primitive that is not registered", {
    prim_neg <- new_primitive(
      "negate",
      function(x) graph_desc_add(self, list(x = x), infer_fn = infer_numeric_uni)[[1L]],
      register = FALSE
    )
    prim_neg[["stablehlo"]] <- prim_negate[["stablehlo"]]
    prim_neg[["vectorize"]] <- rule_vectorize()
    out <- jit(vectorize(prim_neg))(x)
    expect_equal(as_r(out), -as_r(x))
  })

  it("accepts a rule that returns a single output unwrapped", {
    prim_neg <- new_primitive(
      "negate",
      function(x) graph_desc_add(self, list(x = x), infer_fn = infer_numeric_uni)[[1L]],
      register = FALSE
    )
    prim_neg[["stablehlo"]] <- prim_negate[["stablehlo"]]
    prim_neg[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) prim_negate(inputs[[1L]]))
    expect_equal(as_r(jit(vectorize(prim_neg))(x)), -as_r(x))
  })

  it("refuses a declared rule for a primitive whose parameters are not its arguments", {
    prim_scale <- new_primitive(
      "scale",
      function(x, k) {
        graph_desc_add(self, list(x = x), params = list(factor = k), infer_fn = function(x, factor) list(x))[[1L]]
      },
      static = "k",
      register = FALSE
    )
    prim_scale[["vectorize"]] <- rule_vectorize()
    expect_error(jit(vectorize(function(a) prim_scale(a, k = 2)))(x), "cannot call it with them")
  })

  it("checks how many outputs a rule returns", {
    prim_bad <- new_primitive(
      "bad",
      function(x) graph_desc_add(self, list(x = x), infer_fn = infer_numeric_uni)[[1L]],
      register = FALSE
    )
    prim_bad[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
      list(inputs[[1L]], inputs[[1L]])
    })
    expect_error(jit(vectorize(prim_bad))(x), "returned 2 outputs, not 1")
  })

  it("checks what a rule returns", {
    prim_bad <- new_primitive(
      "bad",
      function(x) graph_desc_add(self, list(x = x), infer_fn = infer_numeric_uni)[[1L]],
      register = FALSE
    )
    prim_bad[["vectorize"]] <- rule_vectorize(function(inputs, batched, params, size) {
      list(prim_sum(inputs[[1L]], axes = 1L))
    })
    expect_error(jit(vectorize(prim_bad))(x), "returned a wrong output 1")
  })

  it("rejects an argument without the axis", {
    expect_error(jit(vectorize(function(x) x, in_axes = 3L))(x), "has no axis 3")
  })

  it("rejects mapping over a value that is not an array", {
    expect_error(
      jit(vectorize(function(x) x, in_axes = list(x = 1L)), static = "x")(x = "a"),
      "Can only map over arrays"
    )
  })

  it("rejects a primitive that has no vectorize rule", {
    f <- function(x) prim_qr(x)
    expect_error(
      jit(vectorize(f))(nv_array(1:12, shape = c(3L, 2L, 2L), dtype = "f32")),
      "does not support `prim_qr\\(\\)`"
    )
  })

  it("replays a primitive without a vectorize rule if nothing it reads is mapped over", {
    f <- function(x, m) x + sum(prim_qr(m)[[1L]])
    m <- nv_array(c(2, 1, 1, 3), shape = c(2L, 2L), dtype = "f32")
    out <- jit(vectorize(f, in_axes = list(x = 1L)))(nv_array(c(1, 2), dtype = "f32"), m)
    s <- as_r(jit(function(m) sum(prim_qr(m)[[1L]]))(m))
    expect_equal(as_r(out), array(c(1, 2) + s, dim = 2L), tolerance = 1e-6)
  })
})

describe("rule_vectorize", {
  it("takes either a function or a declaration", {
    expect_error(rule_vectorize(function(...) NULL, scalar = 1L), "either")
    expect_s3_class(rule_vectorize(params = list(axes = param_axes())), "anvl_rule_vectorize")
  })

  it("only takes parameter kinds", {
    expect_error(rule_vectorize(params = list(axes = function(x, size) x)))
  })
})

describe("param_axes", {
  it("shifts axes past the batch axis", {
    expect_identical(param_axes()(c(1L, 3L), size = 5L), c(2L, 4L))
    expect_identical(param_axis_map()(c(2L, 1L), size = 5L), c(1L, 3L, 2L))
    expect_identical(param_shape()(c(2L, 3L), size = 5L), c(5L, 2L, 3L))
    expect_identical(param_per_axis(0L)(c(1L, 2L), size = 5L), c(0L, 1L, 2L))
    expect_identical(param_per_axis(function(size) size)(c(1L, 2L), size = 5L), c(5L, 1L, 2L))
  })
})
