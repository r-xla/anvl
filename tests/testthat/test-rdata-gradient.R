describe("gradient", {
  # An R argument of a jitted call stays open through gradient()'s inline trace:
  # the differentiated body decides which data types the value is used at, exactly
  # as it does under plain jit().

  it("takes a bare R value as an argument", {
    expect_equal(as_array(jit(gradient(function(v) v * v))(2)[[1L]]), 4)
    expect_equal(as_array(jit(gradient(function(v) v * v))(nv_scalar(2))[[1L]]), 4)
    # ... and as a literal in the body of the function being differentiated
    expect_equal(jit(gradient(function(x) x * 2, wrt = "x"))(nv_scalar(1)), list(x = nv_scalar(2)))
  })

  it("reaches an f64 use site exactly through the inline trace", {
    # d/dv of v^2 at f64 is 2v; sqrt(2) is not representable at f32, so this
    # catches a commit at the default dtype.
    f <- function(v) nv_scalar(1, dtype = "f64") * v * v
    r <- jit(gradient(f))(sqrt(2))[[1L]]
    expect_equal(dtype(r), as_dtype("f64"))
    expect_identical(as_array(r), 2 * sqrt(2))
  })

  it("keeps a bare R argument exact in value_and_gradient()", {
    f <- function(v) v * nv_scalar(1, dtype = "f64")
    vg <- jit(value_and_gradient(f))(sqrt(2))
    expect_identical(as_array(vg$value), sqrt(2))
    expect_identical(as_array(vg$grad[[1L]]), 1)
    expect_equal(dtype(vg$value), as_dtype("f64"))
  })

  it("matches plain jit() for an argument used at two data types", {
    # v enters at f64 and, through nv_convert, at f32: the input is supplied at
    # f64 and the f32 use is the program's own convert -- one answer whether the
    # body is differentiated or not.
    f <- function(v) v * nv_scalar(1, dtype = "f64") + nv_convert(nv_convert(v, "f32"), "f64")
    vg <- jit(value_and_gradient(f))(sqrt(2))
    expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
    # both use sites contribute 1 to the gradient, converted losslessly
    expect_identical(as_array(vg$grad[[1L]]), 2)
  })

  it("widens the input past every use when no used data type holds the others", {
    # f16 and bf16 ask for different halves of f32: the gradient's input widens
    # to f32, like a toplevel trace's upload does.
    f <- function(v) nv_convert(nv_convert(v, "f16"), "f32") * nv_convert(nv_convert(v, "bf16"), "f32")
    vg <- jit(value_and_gradient(f))(sqrt(2))
    expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
    expect_equal(dtype(vg$grad[[1L]]), as_dtype("f32"))
  })

  it("matches plain jit() for a body that commits the value at its default", {
    # v * v meets no dtype, so v commits at f32 -- under gradient() exactly as
    # under plain jit().
    f <- function(v) v * v * nv_scalar(1, dtype = "f64")
    vg <- jit(value_and_gradient(f))(sqrt(2))
    expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
  })

  it("shares the argument's value with the enclosing body", {
    q <- function(u) nv_scalar(1, dtype = "f64") * u * u
    # the enclosing trace materializes v at f64 before the gradient call ...
    f <- jit(function(v) {
      w <- v * nv_scalar(1, dtype = "f64")
      w + gradient(q)(v)[[1L]]
    })
    expect_identical(as_array(f(sqrt(2))), sqrt(2) + 2 * sqrt(2))
    # ... and after it, reusing the value the gradient's trace built
    g <- jit(function(v) gradient(q)(v)[[1L]] + v * nv_scalar(1, dtype = "f64"))
    expect_identical(as_array(g(sqrt(2))), 2 * sqrt(2) + sqrt(2))
  })

  it("agrees between two gradient calls on the same bare R argument", {
    q <- function(u) nv_scalar(1, dtype = "f64") * u * u
    f <- jit(function(v) gradient(q)(v)[[1L]] + gradient(q)(v)[[1L]])
    expect_identical(as_array(f(sqrt(2))), 4 * sqrt(2))
  })

  it("keeps a bare R argument exact through a nested gradient", {
    # d/da (2a * a) = 4a, with the inner 2a itself a gradient
    f <- jit(function(v) {
      gradient(function(a) {
        gradient(function(b) nv_scalar(1, dtype = "f64") * b * b)(a)[[1L]] * a
      })(v)[[1L]]
    })
    r <- f(sqrt(2))
    expect_equal(dtype(r), as_dtype("f64"))
    expect_identical(as_array(r), 4 * sqrt(2))
  })

  it("keeps a bare R value outside wrt open too", {
    # n is an R integer used out of its category: it is supplied at i32 and the
    # program converts, like everywhere else.
    f <- jit(gradient(function(x, n) x * nv_convert(n, dtype = "f64"), wrt = "x"))
    expect_identical(as_array(f(nv_scalar(2, dtype = "f64"), 3L)$x), 3)
  })

  it("gives an unused bare R argument an input slot at its default", {
    f <- jit(gradient(function(a, b) nv_scalar(1, dtype = "f64") * a * a))
    r <- f(sqrt(2), pi)
    expect_identical(as_array(r[[1L]]), 2 * sqrt(2))
    expect_identical(as_array(r[[2L]]), 0)
    expect_equal(dtype(r[[2L]]), as_dtype("f32"))
  })

  it("takes an R value written in the enclosing body exactly", {
    # The literal is not an input of any graph: it crosses the gradient boundary
    # and the differentiated body builds it at the data type it meets there.
    f <- jit(function(x) gradient(function(a, b) a * b, wrt = "a")(x, sqrt(2))$a)
    r <- f(nv_scalar(1, dtype = "f64"))
    expect_equal(dtype(r), as_dtype("f64"))
    expect_identical(as_array(r), sqrt(2))
    # ... and asking for its gradient says what a plain R value would.
    g <- jit(function(x) gradient(function(a, b) a * b)(x, sqrt(2)))
    expect_error(g(nv_scalar(1, dtype = "f64")), "passed as a plain R value")
  })

  it("does not bake the R value into the compiled gradient", {
    f <- jit(gradient(function(a, b) nv_scalar(1, dtype = "f64") * a * a))
    expect_identical(as_array(f(sqrt(2), pi)[[1L]]), 2 * sqrt(2))
    # same key, so this is a cache hit -- and it must return its own gradient
    expect_identical(as_array(f(pi, 7)[[1L]]), 2 * pi)
  })
})

# An R value written in the body of a traced function, and one passed as an
# argument, are built into the graph at the data type their use site needs.
# These snapshots pin *how* they are built -- an inlined literal for a scalar, a
# constant for an R array, a convert only out of the value's own category --
# which value-level tests cannot see.
