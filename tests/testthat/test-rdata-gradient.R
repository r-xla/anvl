# An R argument of a jitted call stays open through gradient()'s inline trace:
# the differentiated body decides which dtypes the value is used at, exactly as
# it does under plain jit(). These tests assert concrete numbers (or identity
# with the plain-jit result), so a rounding through the default dtype cannot
# hide behind a tolerance.

test_that("a bare R argument reaches an f64 use site exactly through gradient()", {
  # d/dv of v^2 at f64 is 2v; sqrt(2) is not representable at f32, so this
  # catches a commit at the default dtype.
  f <- function(v) nv_scalar(1, dtype = "f64") * v * v
  r <- jit(gradient(f))(sqrt(2))[[1L]]
  expect_equal(dtype(r), as_dtype("f64"))
  expect_identical(as_array(r), 2 * sqrt(2))
})

test_that("value_and_gradient() keeps a bare R argument exact", {
  f <- function(v) v * nv_scalar(1, dtype = "f64")
  vg <- jit(value_and_gradient(f))(sqrt(2))
  expect_identical(as_array(vg$value), sqrt(2))
  expect_identical(as_array(vg$grad[[1L]]), 1)
  expect_equal(dtype(vg$value), as_dtype("f64"))
})

test_that("a bare R argument used at two dtypes matches plain jit()", {
  # v enters at f64 and, through nv_convert, at f32: the input is supplied at
  # f64 and the f32 use is the program's own convert -- one answer whether the
  # body is differentiated or not.
  f <- function(v) v * nv_scalar(1, dtype = "f64") + nv_convert(nv_convert(v, "f32"), "f64")
  vg <- jit(value_and_gradient(f))(sqrt(2))
  expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
  # both use sites contribute 1 to the gradient, converted losslessly
  expect_identical(as_array(vg$grad[[1L]]), 2)
})

test_that("the input widens past every use when no used dtype holds the others", {
  # f16 and bf16 ask for different halves of f32: the gradient's input widens
  # to f32, like a toplevel trace's upload does.
  f <- function(v) nv_convert(nv_convert(v, "f16"), "f32") * nv_convert(nv_convert(v, "bf16"), "f32")
  vg <- jit(value_and_gradient(f))(sqrt(2))
  expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
  expect_equal(dtype(vg$grad[[1L]]), as_dtype("f32"))
})

test_that("a body that commits the value at its default matches plain jit()", {
  # v * v meets no dtype, so v commits at f32 -- under gradient() exactly as
  # under plain jit().
  f <- function(v) v * v * nv_scalar(1, dtype = "f64")
  vg <- jit(value_and_gradient(f))(sqrt(2))
  expect_identical(as_array(vg$value), as_array(jit(f)(sqrt(2))))
})

test_that("the enclosing body and the gradient body share the argument's value", {
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

test_that("two gradient calls on the same bare R argument agree", {
  q <- function(u) nv_scalar(1, dtype = "f64") * u * u
  f <- jit(function(v) gradient(q)(v)[[1L]] + gradient(q)(v)[[1L]])
  expect_identical(as_array(f(sqrt(2))), 4 * sqrt(2))
})

test_that("a nested gradient keeps a bare R argument exact", {
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

test_that("a bare R value outside wrt stays open too", {
  # n is an R integer used out of its category: it is supplied at i32 and the
  # program converts, like everywhere else.
  f <- jit(gradient(function(x, n) x * nv_convert(n, dtype = "f64"), wrt = "x"))
  expect_identical(as_array(f(nv_scalar(2, dtype = "f64"), 3L)$x), 3)
})

test_that("an unused bare R argument still takes an input slot at its default", {
  f <- jit(gradient(function(a, b) nv_scalar(1, dtype = "f64") * a * a))
  r <- f(sqrt(2), pi)
  expect_identical(as_array(r[[1L]]), 2 * sqrt(2))
  expect_identical(as_array(r[[2L]]), 0)
  expect_equal(dtype(r[[2L]]), as_dtype("f32"))
})

test_that("the compiled gradient does not depend on the R value", {
  f <- jit(gradient(function(a, b) nv_scalar(1, dtype = "f64") * a * a))
  expect_identical(as_array(f(sqrt(2), pi)[[1L]]), 2 * sqrt(2))
  # same key, so this is a cache hit -- and it must return its own gradient
  expect_identical(as_array(f(pi, 7)[[1L]]), 2 * pi)
})

test_that("an R value written in the enclosing body enters gradient() as itself", {
  # A data-carrying RData box is not an input of any graph: it crosses the
  # gradient boundary as the R value, and the body embeds it exactly.
  f <- jit(function(x) {
    v <- maybe_box_arrayish(sqrt(2))
    gradient(function(a, b) a * b, wrt = "a")(x, v)$a
  })
  r <- f(nv_scalar(1, dtype = "f64"))
  expect_equal(dtype(r), as_dtype("f64"))
  expect_identical(as_array(r), sqrt(2))
  # ... and asking for its gradient says what a plain R value would
  g <- jit(function(x) {
    v <- maybe_box_arrayish(sqrt(2))
    gradient(function(a, b) a * b)(x, v)
  })
  expect_error(g(nv_scalar(1, dtype = "f64")), "passed as a plain R value")
})
