f64 <- function(...) nv_array(c(...), dtype = "f64")

describe("jvp", {
  it("returns the derivative of the output along the tangents", {
    f <- function(x, y) sin(x) * y
    out <- jit(jvp(f))(f64(0.5, 1), f64(2, 3), tangents = list(x = f64(1, 2), y = f64(3, 4)))
    xr <- c(0.5, 1)
    yr <- c(2, 3)
    expect_equal(as.vector(as_array(out)), cos(xr) * yr * c(1, 2) + sin(xr) * c(3, 4))
  })

  it("holds the arguments not in wrt fixed", {
    f <- function(x, y) x * y
    expect_equal(jit(jvp(f, wrt = "y"))(f64(2, 3), f64(5, 7), tangents = list(y = f64(1, 1))), f64(2, 3))
    expect_equal(jit(jvp(f, wrt = 1L))(f64(2, 3), f64(5, 7), tangents = list(x = f64(1, 1))), f64(5, 7))
  })

  it("matches the tangents to the arguments by name", {
    f <- function(x, y) x / y
    out <- jit(jvp(f))(y = f64(2), f64(3), tangents = list(y = f64(0), x = f64(1)))
    expect_equal(out, f64(1 / 2))
  })

  it("takes tangents structured like their argument", {
    f <- function(p) p$a * p$b[[1L]]
    out <- jit(jvp(f))(list(a = f64(2), b = list(f64(3))), tangents = list(p = list(a = f64(1), b = list(f64(0)))))
    expect_equal(out, f64(3))
  })

  it("returns tangents structured like the output", {
    f <- function(x) list(sq = x * x, pair = list(x, 2 * x))
    out <- jit(jvp(f))(f64(3), tangents = list(x = f64(1)))
    expect_equal(out, list(sq = f64(6), pair = list(f64(1), f64(2))))
  })

  it("gives an output no tangent reaches a zero tangent", {
    f <- function(x, y) list(x * 2, y, nv_array(1, dtype = "f64"))
    out <- jit(jvp(f, wrt = "x"))(f64(1), f64(4), tangents = list(x = f64(1)))
    expect_equal(out, list(f64(2), f64(0), f64(0)))
  })

  it("gives an output that is not a float a zero tangent", {
    f <- function(x) x > 0
    out <- jit(jvp(f))(f64(-1, 1), tangents = list(x = f64(1, 1)))
    expect_equal(out, nv_array(c(FALSE, FALSE)))
  })

  it("materializes an R value as tangent at the data type of its argument", {
    f <- function(x) x * x
    out <- jit(jvp(f))(nv_array(c(2, 3), dtype = "f32"), tangents = list(x = array(c(1, 0))))
    expect_equal(out, nv_array(c(4, 0), dtype = "f32"))
  })

  it("passes static arguments through", {
    f <- function(x, p) x^p
    g <- jit(jvp(f, wrt = "x"), static = "p")
    expect_equal(g(f64(2), 3L, tangents = list(x = f64(1))), f64(12))
  })

  it("needs no forward rule for what no output depends on", {
    prim_no_forward <- new_primitive(
      "no_forward",
      function(x) graph_desc_add(self, list(x = x), infer_fn = infer_float_uni)[[1L]],
      register = FALSE
    )
    prim_no_forward[["stablehlo"]] <- prim_sin[["stablehlo"]]
    f <- function(x) {
      prim_no_forward(x)
      x * 2
    }
    expect_equal(jit(jvp(f))(f64(1), tangents = list(x = f64(1))), f64(2))
    g <- function(x) prim_no_forward(x)
    expect_error(jit(jvp(g))(f64(1), tangents = list(x = f64(1))), "No forward rule for primitive")
  })

  it("composes with gradient() into second derivatives", {
    f <- function(x) x * x * x * sin(x)
    one <- function() nv_scalar(1, dtype = "f64")
    d1_fwd <- function(x) jvp(f)(x, tangents = list(x = one()))
    x <- nv_scalar(0.7, dtype = "f64")
    xr <- 0.7
    d2 <- 6 * xr * sin(xr) + 6 * xr^2 * cos(xr) - xr^3 * sin(xr)
    # forward over reverse: the product of the Hessian with the tangent
    expect_equal(as.vector(as_array(jit(jvp(gradient(f)))(x, tangents = list(x = one()))$x)), d2)
    expect_equal(as.vector(as_array(jit(gradient(d1_fwd))(x)$x)), d2)
    expect_equal(as.vector(as_array(jit(jvp(d1_fwd))(x, tangents = list(x = one())))), d2)
  })

  it("can only be called inside jit()", {
    expect_error(jvp(sin)(f64(1), tangents = list(x = f64(1))), "inside a .*jit")
  })

  it("refuses a function with an argument named tangents", {
    expect_error(jvp(function(x, tangents) x), "must not have an argument named")
  })

  it("refuses to be called without tangents", {
    expect_error(jit(jvp(function(x) x))(f64(1)), "is missing")
  })

  it("refuses tangents not named by the arguments it differentiates with respect to", {
    f <- function(x, y) x * y
    expect_snapshot(error = TRUE, jit(jvp(f, wrt = "x"))(f64(1), f64(2), tangents = list(y = f64(1))))
    expect_snapshot(error = TRUE, jit(jvp(f))(f64(1), f64(2), tangents = list(x = f64(1))))
    expect_snapshot(error = TRUE, jit(jvp(f, wrt = "x"))(f64(1), f64(2), tangents = f64(1)))
  })

  it("refuses to differentiate with respect to an argument that is not a float array", {
    f <- function(x) x * 2L
    expect_snapshot(
      error = TRUE,
      jit(jvp(f))(nv_array(1L, dtype = "i32"), tangents = list(x = nv_array(1L, dtype = "i32")))
    )
  })

  it("refuses a tangent of another data type or shape than its argument", {
    f <- function(x) x * 2
    expect_snapshot(error = TRUE, jit(jvp(f))(f64(1), tangents = list(x = nv_array(1, dtype = "f32"))))
    expect_snapshot(error = TRUE, jit(jvp(f))(f64(1), tangents = list(x = f64(1, 2))))
  })

  it("refuses a tangent structured unlike its argument", {
    f <- function(p) p$a
    expect_error(
      jit(jvp(f))(list(a = f64(1)), tangents = list(p = f64(1))),
      "structured like the argument"
    )
  })
})

describe("value_and_jvp", {
  it("returns the value of the function and its derivative along the tangents", {
    f <- function(x, y) list(sum = x + y, prod = x * y)
    out <- jit(value_and_jvp(f, wrt = "x"))(f64(2), f64(3), tangents = list(x = f64(1)))
    expect_equal(out, list(value = list(sum = f64(5), prod = f64(6)), tangent = list(sum = f64(1), prod = f64(3))))
  })
})
