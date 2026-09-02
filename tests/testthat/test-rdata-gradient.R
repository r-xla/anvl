describe("gradient", {
  # A value with no data type of its own cannot be differentiated with respect
  # to: the gradient would come back at whatever data type the forward pass
  # settled the value at, so the answer would depend on how the body used it
  # rather than on what the caller passed. Everywhere *else* in a gradient an R
  # value stays open exactly as it does under plain jit() -- as an argument
  # outside `wrt`, and as a literal in the differentiated body.

  it("refuses a value with no data type as the differentiated argument", {
    expect_error(
      jit(gradient(function(v) nv_mean(v * 2)))(array(c(1, 2, 3))),
      "no data type"
    )
    # ... however it reaches the gradient: as the jitted call's own argument,
    expect_error(jit(function(x) gradient(identity)(x))(1), "no data type")
    # or as a literal written where the gradient is taken.
    expect_error(jit(function() gradient(identity)(1))(), "no data type")
  })

  it("takes an argument that has one, at either width", {
    expect_equal(as_array(jit(gradient(function(v) v * v))(nv_scalar(2))[[1L]]), 4)
    expect_equal(
      as_array(jit(gradient(function(v) v * v))(nv_scalar(2, dtype = "f64"))[[1L]]),
      4
    )
    expect_equal(
      jit(gradient(function(x) x * 2, wrt = "x"))(nv_scalar(1)),
      list(x = nv_scalar(2))
    )
  })

  it("keeps a bare R value outside wrt open", {
    # n is an R integer used out of its category: it is supplied at i32 and the
    # program converts, like everywhere else.
    f <- jit(gradient(function(x, n) x * nv_convert(n, dtype = "f64"), wrt = "x"))
    expect_identical(as_array(f(nv_scalar(2, dtype = "f64"), 3L)$x), 3)
  })

  it("keeps a bare R value outside wrt exact through the inline trace", {
    # `k` is an R double used at f64. sqrt(2) is not representable at f32, so a
    # commit at the default would show up in the gradient.
    f <- jit(gradient(function(v, k) v * k, wrt = "v"))
    r <- f(nv_scalar(1, dtype = "f64"), sqrt(2))
    expect_equal(dtype(r$v), as_dtype("f64"))
    expect_identical(as_array(r$v), sqrt(2))
  })

  it("gives an unused bare R argument an input slot at its default", {
    f <- jit(gradient(function(a, b) nv_scalar(1, dtype = "f64") * a * a, wrt = "a"))
    r <- f(nv_scalar(sqrt(2), dtype = "f64"), pi)
    expect_identical(as_array(r$a), 2 * sqrt(2))
  })

  it("does not bake a bare R argument into the compiled gradient", {
    f <- jit(gradient(function(a, b) nv_scalar(1, dtype = "f64") * a * a, wrt = "a"))
    expect_identical(as_array(f(nv_scalar(sqrt(2), dtype = "f64"), pi)$a), 2 * sqrt(2))
    # same key, so this is a cache hit -- and it must return its own gradient
    expect_identical(as_array(f(nv_scalar(pi, dtype = "f64"), 7)$a), 2 * pi)
  })

  it("takes an R value written in the enclosing body exactly", {
    # The literal is not an input of any graph: it crosses the gradient boundary
    # and the differentiated body builds it at the data type it meets there.
    f <- jit(function(x) gradient(function(a, b) a * b, wrt = "a")(x, sqrt(2))$a)
    r <- f(nv_scalar(1, dtype = "f64"))
    expect_equal(dtype(r), as_dtype("f64"))
    expect_identical(as_array(r), sqrt(2))
  })

  it("keeps an f64 argument exact through a nested gradient", {
    # d/da (2a * a) = 4a, with the inner 2a itself a gradient
    f <- jit(function(v) {
      gradient(function(a) {
        gradient(function(b) nv_scalar(1, dtype = "f64") * b * b)(a)[[1L]] * a
      })(v)[[1L]]
    })
    r <- f(nv_scalar(sqrt(2), dtype = "f64"))
    expect_equal(dtype(r), as_dtype("f64"))
    expect_identical(as_array(r), 4 * sqrt(2))
  })

  it("shares an argument's value with the enclosing body", {
    q <- function(u) nv_scalar(1, dtype = "f64") * u * u
    # the enclosing trace uses v before the gradient call ...
    f <- jit(function(v) {
      w <- v * nv_scalar(1, dtype = "f64")
      w + gradient(q)(v)[[1L]]
    })
    v <- nv_scalar(sqrt(2), dtype = "f64")
    expect_identical(as_array(f(v)), sqrt(2) + 2 * sqrt(2))
    # ... and after it
    g <- jit(function(v) gradient(q)(v)[[1L]] + v * nv_scalar(1, dtype = "f64"))
    expect_identical(as_array(g(v)), 2 * sqrt(2) + sqrt(2))
  })

  it("agrees between two gradient calls on the same argument", {
    q <- function(u) nv_scalar(1, dtype = "f64") * u * u
    f <- jit(function(v) gradient(q)(v)[[1L]] + gradient(q)(v)[[1L]])
    expect_identical(as_array(f(nv_scalar(sqrt(2), dtype = "f64"))), 4 * sqrt(2))
  })
})
