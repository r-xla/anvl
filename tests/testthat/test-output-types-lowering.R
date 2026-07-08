# The lowering driver passes each operation's trace-time output types to rules
# that declare an `output_types` parameter; those rules forward them to their
# hlo_* builder so stablehlo skips re-inference. These tests pin the mechanism
# and check it does not change results: a jit-compiled function (which uses
# output_types) must agree with eager execution (which does not go through the
# stablehlo rules).

test_that("a lowering rule accepts and forwards output_types", {
  local_func()
  x <- stablehlo::hlo_input("x", "f32", shape = c(2, 3))
  y <- stablehlo::hlo_input("y", "f32", shape = c(2, 3))
  output_types <- list(at2vt(nv_aval("f32", c(2, 3))))

  out <- prim_add[["stablehlo"]](x, y, output_types = output_types)[[1L]]

  # the builder used the supplied type rather than re-inferring
  expect_equal(out$value_type, output_types[[1L]])
})

test_that("jit lowering with output_types matches eager execution", {
  a <- nv_array(matrix(c(1, 2, 3, 4, 5, 6), 2, 3), dtype = "f32")
  b <- nv_array(matrix(c(6, 5, 4, 3, 2, 1), 2, 3), dtype = "f32")

  # elementwise (binary + unary), reshape, transpose, compare, select
  f <- function(x, y) {
    z <- nv_add(nv_mul(x, y), nv_sub(x, y))
    z <- nv_sqrt(nv_abs(z))
    z <- nv_ifelse(nv_gt(x, y), z, nv_negate(z))
    nv_transpose(nv_reshape(z, c(3, 2)))
  }
  # f32 tolerance: XLA may fuse the jit graph and round differently than
  # per-op eager execution; output_types does not affect the emitted IR.
  expect_equal(as_array(jit(f)(a, b)), as_array(f(a, b)), tolerance = 1e-4)
})

test_that("jit lowering with output_types matches eager for matmul", {
  a <- nv_array(matrix(c(1, 2, 3, 4, 5, 6), 2, 3), dtype = "f32")
  b <- nv_array(matrix(c(1, 0, 0, 1, 1, 1), 3, 2), dtype = "f32")
  f <- function(x, y) nv_matmul(x, y)
  expect_equal(as_array(jit(f)(a, b)), as_array(f(a, b)), tolerance = 1e-4)
})

test_that("output_types is only passed to rules that declare it", {
  # A reduce lowers a sub-graph (its body) via a nested stablehlo() call; the
  # reduce rule does not declare output_types, so nothing leaks into it.
  a <- nv_array(matrix(c(1, 2, 3, 4, 5, 6), 2, 3), dtype = "f32")
  f <- function(x) nv_add(nv_reduce_max(x, dims = 2L), 1)
  expect_equal(as_array(jit(f)(a)), as_array(f(a)), tolerance = 1e-4)
})
