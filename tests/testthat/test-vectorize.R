describe("vectorize", {
  x <- nv_array(matrix(1:6, nrow = 3L), dtype = "f32")

  it("maps over the arguments in args and passes the others to every application", {
    f <- function(x, y) sum(x * y)
    out <- jit(vectorize(f, args = "x"))(x, nv_array(c(1, 2), dtype = "f32"))
    expect_equal(as_r(out), array(c(9, 12, 15), dim = 3L))
  })

  it("maps over all arguments by default", {
    f <- function(x, y) sum(x * y)
    out <- jit(vectorize(f))(x, x)
    expect_equal(as_r(out), array(rowSums(as_r(x)^2), dim = 3L))
  })

  it("takes the arguments in args by position", {
    f <- function(y, x) sum(x * y)
    out <- jit(vectorize(f, args = 2L))(nv_array(c(1, 2), dtype = "f32"), x)
    expect_equal(as_r(out), array(c(9, 12, 15), dim = 3L))
  })

  it("maps over another axis and stacks the outputs along it", {
    f <- function(x) x * 2L
    out <- jit(vectorize(f, axis = 2L))(x)
    expect_equal(as_r(out), as_r(x) * 2)

    g <- function(x) cumsum(x)
    out <- jit(vectorize(g, axis = 2L))(x)
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
    out <- jit(vectorize(f, args = "x"), static = "p")(x, p = 2L)
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
    out <- jit(vectorize(gradient(loss, wrt = "w"), args = "x"))(w, x)
    xr <- as_r(x)
    expect_equal(as_r(out$w), 2 * xr^2 * rep(c(1, 2), each = 3L))
  })

  it("composes with gradient() through indexing", {
    loss <- function(w, x) sum(w[array(c(2L, 1L))] * x)
    w <- nv_array(c(1, 2), dtype = "f32")
    out <- jit(vectorize(gradient(loss, wrt = "w"), args = "x"))(w, x)
    expect_equal(as_r(out$w), as_r(x)[, c(2L, 1L)])
  })

  it("is differentiable", {
    f <- function(w, x) sum(vectorize(function(x) sum(x * w), args = "x")(x)^2L)
    w <- nv_array(c(1, 2), dtype = "f32")
    out <- jit(gradient(f, wrt = "w"))(w, x)
    xr <- as_r(x)
    expect_equal(as_r(out$w), array(2 * colSums(drop(xr %*% c(1, 2)) * xr), dim = 2L))
  })

  it("nests", {
    f <- function(a, b) a * b
    f_ab <- vectorize(vectorize(f, args = "b"), args = "a")
    out <- jit(f_ab)(nv_array(c(1, 2, 3), dtype = "f32"), nv_array(c(10, 20), dtype = "f32"))
    expect_equal(as_r(out), outer(c(1, 2, 3), c(10, 20)))
  })

  it("must be called inside jit", {
    expect_error(vectorize(function(x) x)(x), "inside a")
  })

  it("rejects mapping over an argument that was not passed", {
    expect_error(
      jit(vectorize(function(a, w = a) a * w, args = "w"))(x),
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

  it("rejects arguments that are not formal arguments of f", {
    expect_error(vectorize(function(x) x, args = "y"), "subset of the formal arguments")
  })

  it("rejects arguments of different sizes along the axis", {
    f <- function(x, y) x + y
    expect_error(
      jit(vectorize(f))(x, nv_array(1:4, shape = c(2L, 2L), dtype = "f32")),
      "same size along axis"
    )
  })

  it("rejects an output without room for the batch axis at axis", {
    expect_error(
      jit(vectorize(function(x) sum(x), axis = 2L))(x),
      "room for the batch axis at axis"
    )
  })

  it("keeps the device of an array f closes over", {
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
    expect_error(jit(vectorize(function(x) x, axis = 3L))(x), "must have an axis")
  })

  it("rejects mapping over a value that is not an array", {
    expect_error(jit(vectorize(function(x) x, args = "x"), static = "x")(x = "a"), "Can only map over arrays")
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
    out <- jit(vectorize(f, args = "x"))(nv_array(c(1, 2), dtype = "f32"), m)
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
