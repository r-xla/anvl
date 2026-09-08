test_that("nv_mean", {
  f <- function(y, alpha) {
    mean(y - alpha)
  }

  alpha <- nv_scalar(0.5)
  y <- nv_array(1:10, "f32")
  out <- jit(gradient(f, wrt = "alpha"))(y, alpha)
  expect_equal(shape(out[[1L]]), integer())
})

describe("the default integer", {
  it("does not disturb a gradient that scatters through those indices", {
    # `prim_top_k` and `prim_cummax` route their reverse rule through
    # `prim_scatter` with the forward indices, so a wider index data type has
    # to survive the scatter.
    x <- nv_array(c(3, 1, 4, 1, 5, 9))
    f <- function(x) nv_reduce_sum(nv_top_k(x, k = 3L))
    g <- function(x) nv_reduce_sum(nv_cummax(x))
    at_i32 <- list(
      top_k = as_array(jit(gradient(f))(x)[[1L]]),
      cummax = as_array(jit(gradient(g))(x)[[1L]])
    )
    local_default_dtypes(c(int = "i64"))
    expect_equal(as_array(jit(gradient(f))(x)[[1L]]), at_i32$top_k)
    expect_equal(as_array(jit(gradient(g))(x)[[1L]]), at_i32$cummax)
  })
})
