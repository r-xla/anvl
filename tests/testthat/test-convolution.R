# Reference 2D convolution (NCHW, groups) in base R for parity checks.
.ref_conv2d <- function(x, w, stride = 1L, pad = 0L, groups = 1L) {
  n <- dim(x)[1]; cin <- dim(x)[2]; h <- dim(x)[3]; wd <- dim(x)[4]
  co <- dim(w)[1]; cig <- dim(w)[2]; kh <- dim(w)[3]; kw <- dim(w)[4]
  xp <- array(0, c(n, cin, h + 2 * pad, wd + 2 * pad))
  xp[, , (pad + 1):(pad + h), (pad + 1):(pad + wd)] <- x
  oh <- (h + 2 * pad - kh) %/% stride + 1L
  ow <- (wd + 2 * pad - kw) %/% stride + 1L
  out <- array(0, c(n, co, oh, ow))
  cog <- co %/% groups
  for (oc in seq_len(co)) {
    g <- (oc - 1L) %/% cog
    for (oi in seq_len(oh)) for (oj in seq_len(ow)) {
      acc <- 0
      for (ic in seq_len(cig)) {
        in_c <- g * cig + ic
        for (a in seq_len(kh)) for (b in seq_len(kw)) {
          acc <- acc + xp[1, in_c, (oi - 1L) * stride + a, (oj - 1L) * stride + b] *
            w[oc, ic, a, b]
        }
      }
      out[1, oc, oi, oj] <- acc
    }
  }
  out
}

test_that("nv_conv2d matches a base-R reference (stride, padding, groups)", {
  skip_if_not_installed("pjrt")
  set.seed(1)
  x <- array(rnorm(1 * 3 * 6 * 6), c(1, 3, 6, 6))
  w <- array(rnorm(4 * 3 * 3 * 3), c(4, 3, 3, 3))

  for (cfg in list(
    list(stride = 1L, pad = 1L), list(stride = 2L, pad = 1L),
    list(stride = 1L, pad = 0L)
  )) {
    got <- as.array(nv_conv2d(nv_array(x, dtype = "f32"), nv_array(w, dtype = "f32"),
      stride = cfg$stride, padding = cfg$pad))
    want <- .ref_conv2d(x, w, cfg$stride, cfg$pad)
    expect_equal(dim(got), dim(want))
    expect_lt(max(abs(got - want)), 1e-4)
  }

  # depthwise (groups == in_channels)
  wd <- array(rnorm(3 * 1 * 3 * 3), c(3, 1, 3, 3))
  got <- as.array(nv_conv2d(nv_array(x, dtype = "f32"), nv_array(wd, dtype = "f32"),
    stride = 1L, padding = 1L, groups = 3L))
  want <- .ref_conv2d(x, wd, 1L, 1L, groups = 3L)
  expect_lt(max(abs(got - want)), 1e-4)
})

test_that("nv_conv2d works inside jit()", {
  skip_if_not_installed("pjrt")
  set.seed(2)
  x <- array(rnorm(1 * 2 * 5 * 5), c(1, 2, 5, 5))
  w <- array(rnorm(3 * 2 * 3 * 3), c(3, 2, 3, 3))
  f <- jit(function(a, b) nv_conv2d(a, b, stride = 1L, padding = 1L))
  got <- as.array(f(nv_array(x, dtype = "f32"), nv_array(w, dtype = "f32")))
  want <- .ref_conv2d(x, w, 1L, 1L)
  expect_lt(max(abs(got - want)), 1e-4)
})

test_that("prim_convolution supports asymmetric (causal) padding", {
  skip_if_not_installed("pjrt")
  set.seed(3)
  # 1D causal conv as rank-3: input [N, C, T], kernel [O, I, kT], left-pad only
  x <- array(rnorm(1 * 2 * 6), c(1, 2, 6))
  w <- array(rnorm(3 * 2 * 3), c(3, 2, 3))
  dn <- list(
    input_batch_dimension = 1L, input_feature_dimension = 2L,
    input_spatial_dimensions = 3L,
    kernel_output_feature_dimension = 1L, kernel_input_feature_dimension = 2L,
    kernel_spatial_dimensions = 3L,
    output_batch_dimension = 1L, output_feature_dimension = 2L,
    output_spatial_dimensions = 3L
  )
  got <- as.array(prim_convolution(
    nv_array(x, dtype = "f32"), nv_array(w, dtype = "f32"),
    dimension_numbers = dn, window_strides = 1L,
    padding = matrix(c(2L, 0L), nrow = 1L), lhs_dilation = 1L, rhs_dilation = 1L
  ))
  # causal: output length == input length, out[t] uses inputs <= t
  expect_equal(dim(got), c(1L, 3L, 6L))
})
