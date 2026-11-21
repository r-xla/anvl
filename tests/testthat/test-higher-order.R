# The Jacobian of `f`, a function of one array returning one, at the R array
# `x`, by central differences: of shape c(dim(f(x)), dim(x)).
central_jacobian <- function(f, x, h = 1e-6) {
  f_jit <- jit(f)
  at <- function(v) as_array(f_jit(nv_array(v, dtype = "f64")))
  y <- at(x)
  columns <- vapply(
    seq_along(x),
    function(j) {
      step <- array(0, dim(x))
      step[j] <- h
      as.vector(at(x + step) - at(x - step)) / (2 * h)
    },
    numeric(length(y))
  )
  array(columns, c(dim(y), dim(x)))
}

# `x` with its arrays as R arrays.
as_arrays <- function(x) {
  if (is_anvl_array(x)) {
    as_array(x)
  } else if (is.list(x)) {
    lapply(x, as_arrays)
  } else {
    x
  }
}

describe("jacobian", {
  it("differentiates every element of the value with respect to every element of the arguments", {
    withr::local_seed(1L)
    x <- array(rnorm(6L), c(2L, 3L))
    f <- function(x) x %*% t(x)
    expected <- central_jacobian(f, x)
    out <- jit(jacobian(f))(nv_array(x, dtype = "f64"))
    expect_equal(as_array(out$x), expected, tolerance = 1e-6)
  })

  it("computes the same Jacobian in both modes", {
    withr::local_seed(1L)
    x <- array(runif(6L, 0.5, 2), c(2L, 3L))
    f <- function(x, s) exp(x) * s - x / s
    args <- list(nv_array(x, dtype = "f64"), nv_scalar(1.5, dtype = "f64"))
    backward <- as_arrays(do.call(jit(jacobian(f)), args))
    forward <- as_arrays(do.call(jit(jacobian(f, mode = "forward")), args))
    expect_equal(forward, backward, tolerance = 1e-12)
    # the element of the value at [i, j] depends on that of x at [i, j] alone
    expect_equal(dim(backward$x), c(2L, 3L, 2L, 3L))
    expected <- array(0, c(2L, 3L, 2L, 3L))
    for (i in 1:2) {
      for (j in 1:3) {
        expected[i, j, i, j] <- exp(x[i, j]) * 1.5 - 1 / 1.5
      }
    }
    expect_equal(backward$x, expected, tolerance = 1e-12)
    expect_equal(backward$s, exp(x) + x / 1.5^2, tolerance = 1e-12)
  })

  it("replaces every float array of the value by its derivatives, and gives others NULL", {
    x <- nv_array(c(1, 2), dtype = "f64")
    f <- function(x) list(sq = x^2L, nested = list(sin = sin(x), positive = x > 0))
    for (mode in c("backward", "forward")) {
      out <- as_arrays(jit(jacobian(f, mode = mode))(x))
      expect_equal(out$sq$x, diag(c(2, 4)))
      expect_equal(out$nested$sin$x, diag(cos(c(1, 2))))
      expect_null(out$nested$positive)
    }
  })

  it("structures the derivatives like the arguments in wrt", {
    f <- function(params, scale) (params$w * params$b) * scale
    params <- list(w = nv_array(c(1, 2), dtype = "f64"), b = nv_array(c(3, 4), dtype = "f64"))
    scale <- nv_scalar(2, dtype = "f64")
    for (mode in c("backward", "forward")) {
      out <- as_arrays(jit(jacobian(f, wrt = "params", mode = mode))(params, scale))
      expect_named(out, "params")
      expect_equal(out$params$w, diag(c(6, 8)))
      expect_equal(out$params$b, diag(c(2, 4)))
    }
  })

  it("holds an integer array argument outside wrt fixed", {
    f <- function(x, k) x * nv_convert(k, "f64")
    for (mode in c("backward", "forward")) {
      out <- jit(jacobian(f, wrt = "x", mode = mode))(nv_array(c(1, 2), dtype = "f64"), nv_array(c(2L, 3L)))
      expect_equal(as_array(out$x), diag(c(2, 3)))
    }
  })

  it("gives an array of the value that does not depend on wrt zero derivatives", {
    f <- function(x, y) list(a = x * 2, b = y * 3)
    x <- nv_array(c(1, 2), dtype = "f64")
    y <- nv_array(c(1, 2, 3), dtype = "f64")
    for (mode in c("backward", "forward")) {
      out <- as_arrays(jit(jacobian(f, wrt = "x", mode = mode))(x, y))
      expect_equal(out$a$x, diag(c(2, 2)))
      expect_equal(out$b$x, matrix(0, 3L, 2L))
    }
  })

  it("passes static arguments through", {
    f <- function(x, power) x^power
    g <- jit(jacobian(f, wrt = "x", mode = "forward"), static = "power")
    out <- g(nv_array(c(1, 2), dtype = "f64"), power = 3L)
    expect_equal(as_array(out$x), diag(c(3, 12)))
  })

  it("gives every block the data type of the array of the value", {
    f <- function(x) nv_convert(x, "f64") * 2
    x <- nv_array(c(1, 2), dtype = "f32")
    for (mode in c("backward", "forward")) {
      out <- jit(jacobian(f, mode = mode))(x)
      expect_equal(dtype(out$x), as_dtype("f64"))
      expect_equal(as_array(out$x), diag(c(2, 2)))
    }
  })

  it("gives arrays without elements an empty block", {
    f <- function(x) x^2L
    for (mode in c("backward", "forward")) {
      out <- jit(jacobian(f, mode = mode))(nv_array(numeric(), shape = 0L, dtype = "f64"))
      expect_equal(shape(out$x), c(0L, 0L))
    }
  })

  it("refuses an unknown mode", {
    expect_error(jacobian(function(x) x, mode = "sideways"), "mode")
  })

  it("must be called inside jit", {
    expect_error(jacobian(function(x) x)(nv_array(1)), "inside a .*jit")
  })
})

describe("hessian", {
  it("is the Jacobian of the gradient", {
    x <- c(0.5, 1, 2)
    f <- function(x) sum(x^3L)
    out <- jit(hessian(f))(nv_array(x, dtype = "f64"))
    expect_equal(as_array(out$x$x), diag(6 * x))
  })

  it("holds the second derivatives with respect to two arrays at [[a]][[b]]", {
    f <- function(x, y) sum(x^2L * y)
    x <- c(1, 2)
    y <- c(3, 4)
    out <- as_arrays(jit(hessian(f))(nv_array(x, dtype = "f64"), nv_array(y, dtype = "f64")))
    expect_equal(out$x$x, diag(2 * y))
    expect_equal(out$x$y, diag(2 * x))
    expect_equal(out$y$x, diag(2 * x))
    expect_equal(out$y$y, matrix(0, 2L, 2L))
  })

  it("gives the block of a and b the shape of a followed by that of b", {
    f <- function(x, y) sum(x^3L) + sum(y^2L)
    out <- as_arrays(jit(hessian(f))(nv_array(c(1, 2), dtype = "f64"), nv_array(c(1, 2, 3), dtype = "f64")))
    expect_equal(out$x$x, diag(c(6, 12)))
    expect_equal(out$x$y, matrix(0, 2L, 3L))
    expect_equal(out$y$x, matrix(0, 3L, 2L))
    expect_equal(out$y$y, diag(2, 3L))
  })

  it("differentiates with respect to the arguments in wrt", {
    f <- function(x, y) sum(x^2L * y)
    out <- jit(hessian(f, wrt = "x"))(nv_array(c(1, 2), dtype = "f64"), nv_array(c(3, 4), dtype = "f64"))
    expect_named(out, "x")
    expect_named(out$x, "x")
    expect_equal(as_array(out$x$x), diag(c(6, 8)))
  })

  it("must be called inside jit", {
    expect_error(hessian(function(x) sum(x))(nv_array(1)), "hessian")
  })
})
