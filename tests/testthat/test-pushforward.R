f64 <- function(...) nv_array(c(...), dtype = "f64")

describe("pushforward", {
  x <- f64(0.5, 1)
  y <- f64(2, 3)

  it("pushes a tangent of the arguments forward to the value", {
    g <- jit(function(x, y, dx, dy) pushforward(function(x, y) sin(x) * y, x, y)(dx, dy))
    out <- g(x, y, f64(1, 2), f64(3, 4))
    expect_equal(as.numeric(out), cos(c(0.5, 1)) * c(2, 3) * c(1, 2) + sin(c(0.5, 1)) * c(3, 4))
  })

  it("matches the tangents to the formals of .f as in a call, holding the others fixed", {
    f <- function(x, y) x * y
    g <- jit(function(x, y) pushforward(f, y = y, x)(y = f64(1, 1)))
    expect_equal(g(x, y), x)
  })

  it("matches tangents to arguments passed through ... by position, and by name where they have one", {
    f <- function(...) ..1 * ..2
    expect_equal(jit(function(x, y) pushforward(f, x, y)(f64(1, 1), f64(0, 0)))(x, y), y)
    expect_equal(jit(function(x, y) pushforward(f, x, y)(NULL, f64(1, 1)))(x, y), x)
    g <- function(x, ...) x * list(...)$b
    expect_equal(jit(function(x, y) pushforward(g, x, b = y)(f64(1, 1)))(x, y), y)
    expect_equal(jit(function(x, y) pushforward(g, x, b = y)(b = f64(1, 1)))(x, y), x)
  })

  it("refuses a tangent of the wrong shape for an argument passed through ...", {
    f <- function(...) ..1 * ..2
    expect_error(
      jit(function(x, y) pushforward(f, x, y)(f64(1, 1), f64(1)))(x, y),
      "must have the data type and shape of the argument"
    )
  })

  it("takes tangents structured like their argument, with NULL for any part", {
    f <- function(p) p$a * p$b[[1L]] + p$b[[2L]]
    g <- jit(function(a, b1, b2) pushforward(f, list(a = a, b = list(b1, b2)))(list(a = f64(1), b = NULL)))
    expect_equal(g(f64(2), f64(3), f64(4)), f64(3))
  })

  it("returns tangents structured like the value", {
    f <- function(x) list(sq = x * x, pair = list(x, 2 * x))
    out <- jit(function(x) pushforward(f, x)(f64(1)))(f64(3))
    expect_equal(out, list(sq = f64(6), pair = list(f64(1), f64(2))))
  })

  it("gives a value no tangent reaches a zero tangent", {
    f <- function(x, y) list(x * 2, y, f64(1))
    out <- jit(function(x, y) pushforward(f, x, y)(x = f64(1)))(f64(1), f64(4))
    expect_equal(out, list(f64(2), f64(0), f64(0)))
  })

  it("gives a value that is not a float no tangent", {
    f <- function(x) list(sq = x * x, pos = x > 0)
    pos <- "unset"
    g <- jit(function(x) {
      out <- pushforward(f, x)(f64(1, 1))
      pos <<- out$pos
      out$sq
    })
    expect_equal(g(x), 2 * x)
    expect_null(pos)
  })

  it("materializes an R value as tangent at the data type of its argument", {
    out <- jit(function(x) pushforward(\(x) x * x, x)(array(c(1, 0))))(nv_array(c(2, 3), dtype = "f32"))
    expect_equal(out, nv_array(c(4, 0), dtype = "f32"))
  })

  it("holds an argument that is not a float array fixed", {
    f <- function(x, n, s) x * nv_convert(n, "f64") * s
    g <- jit(function(x, n) pushforward(f, x, n, 2)(f64(1, 1)))
    expect_equal(g(x, nv_array(c(1L, 2L))), f64(2, 4))
  })

  it("passes static arguments through", {
    g <- jit(function(x, p, v) pushforward(function(x, p) x^p, x, p)(v), static = "p")
    expect_equal(g(f64(2), 3L, f64(1)), f64(12))
  })

  it("can be called more than once, e.g. for the columns of a Jacobian", {
    jacobian <- jit(function(x) {
      # only elementwise primitives have a pushforward rule so far
      push <- pushforward(\(x) nv_sin(x) * x, x)
      eye <- nv_eye(3L, dtype = "f64")
      columns <- lapply(1:3, \(j) nv_reshape(push(eye[, j]), c(3L, 1L)))
      do.call(nv_concatenate, c(columns, list(axis = 2L)))
    })
    xs <- c(1, 2, 3)
    expected <- diag(sin(xs) + xs * cos(xs))
    expect_equal(as_array(jacobian(f64(1, 2, 3))), expected)
  })

  it("can be called in a function traced inside the one it was made in", {
    g <- jit(function(x, p) {
      push <- pushforward(\(x) x * x, x)
      nv_if(p, \() push(f64(1, 1)), \() x)
    })
    expect_equal(g(x, nv_scalar(TRUE)), 2 * x)
  })

  it("needs no pushforward rule for what no output depends on", {
    prim_no_pushforward <- new_primitive(
      "no_pushforward",
      function(x) graph_desc_add(self, list(x = x), infer_fn = infer_float_uni)[[1L]],
      register = FALSE
    )
    prim_no_pushforward[["stablehlo"]] <- prim_sin[["stablehlo"]]
    f <- function(x) {
      prim_no_pushforward(x)
      x * 2
    }
    expect_equal(jit(function(x) pushforward(f, x)(f64(1)))(f64(1)), f64(2))
    g <- function(x) prim_no_pushforward(x)
    expect_error(jit(function(x) pushforward(g, x)(f64(1)))(f64(1)), "No pushforward rule for primitive")
  })

  it("composes with gradient() into second derivatives", {
    f <- function(x) x * x * x * sin(x)
    one <- function() nv_scalar(1, dtype = "f64")
    d1_rev <- function(x) gradient(f)(x)$x
    d1_fwd <- function(x) pushforward(f, x)(one())
    xr <- 0.7
    d2 <- 6 * xr * sin(xr) + 6 * xr^2 * cos(xr) - xr^3 * sin(xr)
    x <- nv_scalar(xr, dtype = "f64")
    # forward over reverse: the product of the Hessian with the tangent
    expect_equal(as.numeric(jit(function(x) pushforward(d1_rev, x)(one()))(x)), d2)
    expect_equal(as.numeric(jit(gradient(d1_fwd))(x)$x), d2)
    expect_equal(as.numeric(jit(function(x) pushforward(d1_fwd, x)(one()))(x)), d2)
  })

  it("refuses to be called outside jit()", {
    expect_error(pushforward(\(x) x, x), "can only be called inside")
  })

  it("refuses a pushforward called outside the function pushforward() was called in", {
    stash <- NULL
    jit(function(x) {
      stash <<- pushforward(\(x) x * x, x)
      x
    })(x)
    expect_error(jit(function(x) stash(x))(x), "can only be called in the function")
  })

  it("refuses a tangent for an argument it was not evaluated at", {
    f <- function(x, y = 1) x * y
    expect_error(
      jit(function(x) pushforward(f, x)(y = f64(1)))(f64(1)),
      "only of the arguments"
    )
  })

  it("refuses a tangent for an array that is not a float", {
    f <- function(x, n) x * nv_convert(n, "f64")
    expect_snapshot(error = TRUE, jit(function(x, n) pushforward(f, x, n)(n = n))(f64(1), nv_array(1L, dtype = "i32")))
  })

  it("refuses a tangent of another data type, shape or structure than its argument", {
    f <- function(x) x * 2
    expect_snapshot(error = TRUE, jit(function(x) pushforward(f, x)(nv_array(1, dtype = "f32")))(f64(1)))
    expect_snapshot(error = TRUE, jit(function(x) pushforward(f, x)(f64(1, 2)))(f64(1)))
    g <- function(p) p$a
    expect_error(
      jit(function(a) pushforward(g, list(a = a))(f64(1)))(f64(1)),
      "structured like the argument"
    )
  })
})

describe("value_and_pushforward", {
  it("returns the value of the function together with its pushforward", {
    f <- function(x, y) list(sum = x + y, prod = x * y)
    g <- jit(function(x, y) {
      out <- value_and_pushforward(f, x, y)
      list(value = out$value, tangent = out$pushforward(x = f64(1)))
    })
    expect_equal(
      g(f64(2), f64(3)),
      list(value = list(sum = f64(5), prod = f64(6)), tangent = list(sum = f64(1), prod = f64(3)))
    )
  })

  it("returns exactly the value and the pushforward", {
    names_of <- NULL
    jit(function(x) {
      names_of <<- names(value_and_pushforward(function(x) x * x, x))
      x
    })(f64(1))
    expect_identical(names_of, c("value", "pushforward"))
  })
})
