test_that("an R value takes the dtype of the array it meets", {
  f <- function(x) x * 1L
  expect_equal(
    jit(f)(nv_scalar(2L, "i16")),
    nv_scalar(2L, dtype = "i16")
  )
  expect_equal(
    jit(f)(nv_scalar(2, "f64")),
    nv_scalar(2, dtype = "f64")
  )
})

test_that("an R value that meets nothing commits to the default dtype", {
  expect_equal(jit(function() 1 * 2)(), nv_scalar(2, dtype = default_float()))
  expect_equal(jit(function() 1L * 2L)(), nv_scalar(2L, dtype = default_int()))
})

test_that("a value that has committed keeps its dtype", {
  # `x * 1L` commits to i32 -- the R value cannot stay a bool -- and the i16
  # then promotes against a real i32, which wins.
  f <- function(x, y) (x * 1L) + y
  expect_equal(
    jit(f)(nv_scalar(TRUE), nv_scalar(2L, "i16")),
    nv_scalar(3L, dtype = default_int())
  )
})

test_that("prim_if outputs keep the dtype of their branches", {
  f <- function(pred, x) {
    x <- x * 2L
    nv_if(pred, \() x, \() x * x) * nv_scalar(3L, dtype = "i16")
  }
  expect_equal(
    jit(f)(nv_scalar(TRUE), nv_scalar(TRUE)),
    nv_scalar(6L, dtype = default_int())
  )
})

test_that("prim_while carries the dtype of its state", {
  f <- jit(function(n) {
    i <- 0L * nv_scalar(TRUE)
    nv_while(list(i = i), \(i) i <= n, \(i) {
      i <- i + 1L
      list(i = i)
    })[[1L]] *
      nv_scalar(3L, dtype = "i16")
  })
  expect_equal(
    f(nv_scalar(10L)),
    nv_scalar(33L, dtype = default_int())
  )
})

test_that("a logical R value is a bool, not an uncommitted value", {
  f <- function(x) x * TRUE
  graph <- trace_fn(f, list(x = nv_scalar(1L)))
  # The logical is built at `bool` -- the only dtype that holds it faithfully --
  # and the program converts it to the dtype it met, so the multiply itself sees
  # an i32.
  mul <- Filter(function(call) call$primitive$name == "mul", graph$calls)[[1L]]
  expect_equal(dtype(mul$inputs[[2L]]$aval), default_int())
})

describe("eager/jit equivalence", {
  it("agrees for promote_like() and promote_common()", {
    expect_eager_jit_equal_grid(list(
      "anchored promotion" = function(x, v) {
        args <- as_anvl_arrays(x = x, v = v, .promote = promote_like("x"))
        args$x * args$v
      },
      "promotion to the common dtype" = function(x, v) {
        args <- as_anvl_arrays(x, v, .promote = promote_common())
        args[[1L]] * args[[2L]]
      }
    ))
  })
})

describe("the default float", {
  it("does not change the yielding rule", {
    local_default_dtypes(c(float = "f64"))
    expect_equal(dtype(nv_array(1, dtype = "f32") + 1.5), as_dtype("f32"))
    expect_equal(dtype(jit(function(x) x * 2)(nv_array(1, dtype = "f32"))), as_dtype("f32"))
    # Crossing a category takes the *default* of the other category.
    expect_equal(dtype(nv_array(1L, dtype = "i32") + 1.5), as_dtype("f64"))
  })
})

describe("the default integer", {
  it("does not change the yielding rule", {
    local_default_dtypes(c(int = "i64"))
    expect_equal(dtype(nv_array(1L, dtype = "i32") + 1L), as_dtype("i32"))
    expect_equal(dtype(nv_array(1L, dtype = "i8") * 2L), as_dtype("i8"))
    expect_equal(dtype(nv_array(TRUE) + 1L), as_dtype("i64"))
  })
})

describe("eager code", {
  it("reads the same default the operation runs with", {
    skip_if_no_quickr()
    # A plain R helper decides a promotion eagerly, between dispatches. The
    # default it reads is the one of the backend in force, which is also the
    # backend the operation then runs on.
    promote <- function(x) as_anvl_arrays(x, 1.5, .promote = promote_common())[[2L]]
    expect_equal(dtype(promote(nv_array(1L, dtype = "i32"))), default_float())
    with_backend("quickr", {
      expect_equal(dtype(promote(nv_array(1L, dtype = "i32"))), as_dtype("f64"))
      expect_equal(peek_dtype(1.5), as_dtype("f64"))
      expect_equal(dtype(nv_fill(0, 3)), as_dtype("f64"))
    })
  })
})
