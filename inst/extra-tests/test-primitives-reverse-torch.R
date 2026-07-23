build_extra_args <- function(args_f, shp, dtype) {
  if (is.null(args_f)) {
    return(list(list(), list()))
  }
  args_f(shp, dtype)
}

wrap_uni_anvl <- function(.f, args_anvl, shp) {
  if (identical(shp, integer())) {
    return(\(x) {
      do.call(.f, c(list(x), args_anvl))
    })
  }

  \(x) {
    res <- do.call(.f, c(list(x), args_anvl))
    nv_reduce_sum(res, axes = seq_along(shape(res)), drop = TRUE)
  }
}

wrap_uni_torch <- function(.g, args_torch, shp) {
  if (identical(shp, integer())) {
    return(\(x) {
      do.call(.g, c(list(x), args_torch))
    })
  }

  \(x) {
    res <- do.call(.g, c(list(x), args_torch))
    torch::torch_sum(res, dim = seq_along(res$shape), keepdim = FALSE)
  }
}

wrap_biv_anvl <- function(.f, args_anvl, shp) {
  if (identical(shp, integer())) {
    return(\(lhs, rhs) {
      do.call(.f, c(list(lhs, rhs), args_anvl))
    })
  }

  \(lhs, rhs) {
    x <- do.call(.f, c(list(lhs, rhs), args_anvl))
    nv_reduce_sum(x, axes = seq_along(shape(x)), drop = TRUE)
  }
}

wrap_biv_torch <- function(.g, args_torch, shp) {
  \(lhs, rhs) {
    x <- do.call(.g, c(list(lhs, rhs), args_torch))
    num_remaining <- length(shp)
    if (num_remaining > 0L) torch::torch_sum(x, dim = seq_len(num_remaining)) else x
  }
}

verify_grad_uni_scalar <- function(
  .f,
  .g,
  naxes = 0L,
  dtypes = "f32",
  args_f = NULL,
  tol = 1e-5,
  non_negative = FALSE,
  gen = NULL
) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()

  if (is.null(gen)) {
    x <- generate_test_data(integer(), dtype, non_negative = non_negative)
  } else {
    x <- gen(shp, dtype)
  }

  x_anvl <- nv_scalar(x, dtype = dtype)

  # I think there is a bug in torch, so we can't use torch_scalar_tensor
  x_torch <- torch::torch_scalar_tensor(x, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))
  x_torch$retain_grad()

  args <- build_extra_args(args_f, shp, dtype)
  args_anvl <- args[[1L]]
  args_torch <- args[[2L]]

  .f_anvl <- \(x) {
    do.call(.f, c(list(x), args_anvl))
  }
  .g_torch <- \(x) {
    do.call(.g, c(list(x), args_torch))
  }

  grads_anvl <- jit(gradient(.f_anvl))(x_anvl)
  out <- .g_torch(x_torch)
  out$backward(retrain_graph = TRUE)

  expect_equal(to_abstract(grads_anvl[[1L]], TRUE), to_abstract(x_anvl, TRUE))

  testthat::expect_equal(
    tengen::as_array(grads_anvl[[1L]]),
    as_array_torch(x_torch$grad),
    tolerance = tol
  )
}

verify_grad_uni_tensor <- function(
  .f,
  .g,
  naxes = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  shape = NULL,
  tol = 1e-5,
  non_negative = FALSE,
  gen = NULL
) {
  shp <- if (is.null(shape)) sample(1:3, naxes, replace = TRUE) else shape
  dtype <- sample(dtypes, 1L)

  if (is.null(gen)) {
    x <- array(
      generate_test_data(shp, dtype = dtype, non_negative = non_negative),
      shp
    )
  } else {
    x <- gen(shp, dtype)
  }

  x_anvl <- nv_array(x, dtype = dtype)

  x_torch <- torch::torch_tensor(
    x,
    requires_grad = TRUE,
    dtype = str_to_torch_dtype(dtype)
  )

  args <- build_extra_args(args_f, shp, dtype)
  args_anvl <- args[[1L]]
  args_torch <- args[[2L]]

  .f_anvl <- wrap_uni_anvl(.f, args_anvl, shp)
  .g_torch <- wrap_uni_torch(.g, args_torch, shp)

  grads_anvl <- jit(gradient(.f_anvl))(x_anvl)
  .g_torch(x_torch)$backward()

  expect_equal(to_abstract(grads_anvl[[1L]], TRUE), to_abstract(x_anvl, TRUE))

  testthat::expect_equal(
    tengen::as_array(grads_anvl[[1L]]),
    as_array_torch(x_torch$grad),
    tolerance = tol
  )
}

verify_grad_biv_scalar <- function(
  .f,
  .g,
  naxes = 0L,
  dtypes = "f32",
  args_f = NULL,
  tol = 1e-5,
  non_negative = list(FALSE, FALSE),
  gen_lhs = NULL,
  gen_rhs = NULL
) {
  dtype <- sample(dtypes, 1L)
  shp <- integer()

  if (length(non_negative) < 2) {
    non_negative <- rep(non_negative, 2)
  }

  if (is.null(gen_lhs)) {
    lhs <- generate_test_data(integer(), dtype, non_negative = non_negative[[1]])
  } else {
    lhs <- gen_lhs(shp, dtype)
  }
  if (is.null(gen_rhs)) {
    rhs <- generate_test_data(integer(), dtype, non_negative = non_negative[[2]])
  } else {
    rhs <- gen_rhs(shp, dtype)
  }

  lhs_anvl <- nv_scalar(lhs, dtype = dtype)
  rhs_anvl <- nv_scalar(rhs, dtype = dtype)

  # I think there is a bug in torch, so we can't use torch_scalar_tensor
  lhs_torch <- torch::torch_scalar_tensor(lhs, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))
  lhs_torch$retain_grad()
  rhs_torch <- torch::torch_scalar_tensor(rhs, requires_grad = TRUE, dtype = str_to_torch_dtype(dtype))
  rhs_torch$retain_grad()

  args <- build_extra_args(args_f, shp, dtype)
  args_anvl <- args[[1L]]
  args_torch <- args[[2L]]

  .f_anvl <- \(lhs, rhs) {
    do.call(.f, c(list(lhs, rhs), args_anvl))
  }
  .g_torch <- \(lhs, rhs) {
    do.call(.g, c(list(lhs, rhs), args_torch))
  }

  grads_anvl <- jit(gradient(.f_anvl))(lhs_anvl, rhs_anvl)
  out <- .g_torch(lhs_torch, rhs_torch)
  out$backward(retrain_graph = TRUE)

  expect_equal(to_abstract(grads_anvl[[1L]], TRUE), to_abstract(lhs_anvl, TRUE))
  expect_equal(to_abstract(grads_anvl[[2L]], TRUE), to_abstract(rhs_anvl, TRUE))

  testthat::expect_equal(
    tengen::as_array(grads_anvl[[1L]]),
    as_array_torch(lhs_torch$grad), # nolint
    tolerance = tol
  )

  testthat::expect_equal(
    tengen::as_array(grads_anvl[[2L]]),
    as_array_torch(rhs_torch$grad), # nolint
    tolerance = tol
  )
}

verify_grad_biv_tensor <- function(
  .f,
  .g,
  naxes = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  shape = NULL,
  tol = 1e-5,
  non_negative = list(FALSE, FALSE),
  gen_lhs = NULL,
  gen_rhs = NULL
) {
  # Prefer shapes without size-0 or size-1 axes to avoid backend broadcast edge-cases
  shp <- if (is.null(shape)) sample(1:3, naxes, replace = TRUE) else shape
  dtype <- sample(dtypes, 1)

  if (length(non_negative) < 2) {
    non_negative <- rep(non_negative, 2)
  }

  if (is.null(gen_lhs)) {
    lhs <- array(
      generate_test_data(shp, dtype = dtype, non_negative = non_negative[[1]]), # nolint
      shp
    )
  } else {
    lhs <- gen_lhs(shp, dtype)
  }
  if (is.null(gen_rhs)) {
    rhs <- array(
      generate_test_data(shp, dtype = dtype, non_negative = non_negative[[2]]), # nolint
      shp
    )
  } else {
    rhs <- gen_rhs(shp, dtype)
  }

  lhs_anvl <- nv_array(lhs)
  rhs_anvl <- nv_array(rhs)

  lhs_torch <- torch::torch_tensor(lhs, requires_grad = TRUE)
  rhs_torch <- torch::torch_tensor(rhs, requires_grad = TRUE)

  args <- build_extra_args(args_f, shp, dtype)
  args_anvl <- args[[1L]]
  args_torch <- args[[2L]]

  .f_anvl <- wrap_biv_anvl(.f, args_anvl, shp)
  .g_torch <- wrap_biv_torch(.g, args_torch, shp)

  grads_anvl <- jit(gradient(.f_anvl))(lhs_anvl, rhs_anvl)
  .g_torch(lhs_torch, rhs_torch)$backward()

  expect_equal(to_abstract(grads_anvl[[1L]], TRUE), to_abstract(lhs_anvl, TRUE))
  expect_equal(to_abstract(grads_anvl[[2L]], TRUE), to_abstract(rhs_anvl, TRUE))

  testthat::expect_equal(
    tengen::as_array(grads_anvl[[1L]]),
    as_array_torch(lhs_torch$grad), # nolint
    tolerance = tol # nolint
  )

  testthat::expect_equal(
    tengen::as_array(grads_anvl[[2L]]),
    as_array_torch(rhs_torch$grad),
    tolerance = tol
  )
}

verify_grad_biv <- function(
  f,
  g,
  naxes = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  tol = 1e-5,
  non_negative = list(FALSE, FALSE),
  gen_lhs = NULL,
  gen_rhs = NULL
) {
  verify_grad_biv_scalar(
    f,
    g,
    naxes = 0L,
    dtypes = dtypes,
    args_f = args_f,
    tol = tol,
    non_negative = non_negative,
    gen_lhs = gen_lhs,
    gen_rhs = gen_rhs
  )
  verify_grad_biv_tensor(
    f,
    g,
    naxes = naxes,
    dtypes = dtypes,
    args_f = args_f,
    tol = tol,
    non_negative = non_negative,
    gen_lhs = gen_lhs,
    gen_rhs = gen_rhs
  )
}

verify_grad_uni <- function(
  f,
  g,
  naxes = sample(1:3, 1L),
  dtypes = "f32",
  args_f = NULL,
  tol = 1e-5,
  non_negative = FALSE,
  skip_scalar = FALSE,
  gen = NULL
) {
  if (!skip_scalar) {
    verify_grad_uni_scalar(
      f,
      g,
      naxes = 0L,
      dtypes = dtypes,
      args_f = args_f,
      tol = tol,
      non_negative = non_negative,
      gen = gen
    )
  }
  verify_grad_uni_tensor(
    f,
    g,
    naxes = naxes,
    dtypes = dtypes,
    args_f = args_f,
    tol = tol,
    non_negative = non_negative,
    gen = gen
  )
}


test_that("prim_add", {
  verify_grad_biv(prim_add, torch::torch_add)
})

test_that("prim_sub", {
  verify_grad_biv(prim_sub, torch::torch_sub)
})

test_that("prim_mul", {
  verify_grad_biv(prim_mul, torch::torch_mul)
})

test_that("prim_negate", {
  verify_grad_uni(prim_negate, torch::torch_neg)
})

test_that("prim_exp", {
  withr::local_seed(12)
  verify_grad_uni(prim_exp, torch::torch_exp, tol = 1e-4)
})

test_that("prim_log", {
  verify_grad_uni(prim_log, torch::torch_log)
})

test_that("prim_div", {
  # TODO:
  # Need to determine what to do with non-differentiable values:
  # https://docs.pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions
  verify_grad_biv(prim_div, torch::torch_div)
})

test_that("prim_pow", {
  # TODO:
  # Need to determine what to do with non-differentiable values:
  # https://docs.pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions
  verify_grad_biv(prim_pow, torch::torch_pow, non_negative = list(TRUE, FALSE), tol = 1e-5)
  verify_grad_biv(prim_pow, torch::torch_pow, non_negative = list(FALSE, TRUE), tol = 1e-5)
})

test_that("prim_reduce_sum", {
  x_arr <- array(1:6, c(2, 3))
  x <- nv_array(x_arr, dtype = "f32")
  f <- function(a) {
    y <- prim_reduce_sum(a, axes = 2L, drop = TRUE)
    prim_reduce_sum(y, axes = 1L, drop = TRUE)
  }
  grads <- jit(gradient(f))(x)
  expect_equal(tengen::as_array(grads[[1L]]), array(1, dim = c(2, 3)))
  # TODO: Also test with drop = FALSE
  f <- function(a) {
    y <- prim_reduce_sum(a, axes = 2L, drop = FALSE)
    prim_reduce_sum(y, axes = 1:2, drop = TRUE)
  }
  grads <- jit(gradient(f))(x)
  expect_equal(tengen::as_array(grads[[1L]]), array(1, dim = c(2, 3)))
})

test_that("prim_transpose", {
  verify_grad_uni_tensor(prim_transpose, \(x, permutation) x$permute(permutation), naxes = 3L, args_f = \(shp, dtype) {
    axes <- sample(seq_along(shp))
    list(
      list(permutation = axes),
      list(permutation = axes)
    )
  })
})

describe("prim_cumsum", {
  it("vector gradient", {
    verify_grad_uni_tensor(
      prim_cumsum,
      torch::torch_cumsum,
      shape = 5L,
      args_f = \(shp, dtype) list(list(axis = 1L), list(axis = 1L))
    )
  })
  it("matrix gradient along each axis", {
    for (d in 1:2) {
      verify_grad_uni_tensor(
        prim_cumsum,
        torch::torch_cumsum,
        shape = c(3L, 4L),
        args_f = local({
          dl <- d
          \(shp, dtype) list(list(axis = dl), list(axis = dl))
        })
      )
    }
  })
})

test_that("prim_broadcast_in_axes", {
  input_shape <- c(2L, 1L, 3L)
  target_shape <- c(4L, 2L, 5L, 3L)

  f <- function(x, shape) {
    res <- nv_broadcast_to(x, shape)
    nv_reduce_sum(res, axes = seq_along(shape), drop = TRUE)
  }

  verify_grad_uni_tensor(
    nv_broadcast_to,
    \(x, shape) x$broadcast_to(shape),
    shape = input_shape,
    args_f = \(shp, dtype) {
      list(
        list(shape = target_shape),
        list(shape = target_shape)
      )
    }
  )
})

test_that("prim_ifelse", {
  shp <- c(2L, 3L)
  x_arr <- generate_test_data(shp, dtype = "bool")
  x_anvl <- nv_array(x_arr, dtype = "bool")
  x_torch <- torch::torch_tensor(x_arr, dtype = torch::torch_bool())

  a_arr <- generate_test_data(shp, dtype = "f32")
  b_arr <- generate_test_data(shp, dtype = "f32")
  a_anvl <- nv_array(a_arr, dtype = "f32")
  b_anvl <- nv_array(b_arr, dtype = "f32")
  a_torch <- torch::torch_tensor(a_arr, requires_grad = TRUE, dtype = torch::torch_float32())
  b_torch <- torch::torch_tensor(b_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  f_anvl <- function(a, b) {
    out <- prim_ifelse(x_anvl, a, b)
    nv_reduce_sum(out, axes = 1:2, drop = TRUE)
  }
  grads <- jit(gradient(f_anvl))(a_anvl, b_anvl)

  out_t <- torch::torch_where(x_torch, a_torch, b_torch)
  torch::torch_sum(out_t)$backward()

  expect_equal(tengen::as_array(grads[[1L]]), as_array_torch(a_torch$grad), tolerance = 1e-6)
  expect_equal(tengen::as_array(grads[[2L]]), as_array_torch(b_torch$grad), tolerance = 1e-6)
})

test_that("prim_reshape", {
  in_shape <- c(2L, 3L)
  out_shape <- c(3L, 2L)
  verify_grad_uni_tensor(
    prim_reshape,
    function(x, shape) x$reshape(shape),
    shape = in_shape,
    args_f = function(shp, dtype) list(list(shape = out_shape), list(shape = out_shape))
  )
})

test_that("prim_convert", {
  target_dtype <- "f64"
  verify_grad_uni_tensor(
    \(x, dtype) prim_convert(x, dtype = dtype, ambiguous = FALSE),
    function(x, dtype) x$to(dtype = dtype),
    dtypes = "f32",
    args_f = function(shp, dtype) {
      list(
        list(dtype = target_dtype),
        list(dtype = torch::torch_float64())
      )
    }
  )
})

test_that("prim_sqrt", {
  verify_grad_uni(prim_sqrt, torch::torch_sqrt, non_negative = TRUE, tol = 1e-5)
})

test_that("prim_rsqrt", {
  # f64 to avoid log message
  verify_grad_uni(prim_rsqrt, torch::torch_rsqrt, non_negative = TRUE, tol = 1e-5, dtypes = "f64")
})

test_that("prim_tanh", {
  withr::local_seed(12)
  verify_grad_uni(prim_tanh, torch::torch_tanh, tol = 1e-4)
})

test_that("prim_tan", {
  # values near pi/2 cause divergence -> avoid unlucky seed
  withr::local_seed(12)
  verify_grad_uni_tensor(
    prim_tan,
    torch::torch_tan,
    tol = 1e-4
  )
})

test_that("prim_sin", {
  verify_grad_uni(prim_sin, torch::torch_sin, tol = 1e-5)
})

test_that("prim_cos", {
  verify_grad_uni(prim_cos, torch::torch_cos, tol = 1e-5)
})

# CHLO ops: inverse trig, hyperbolic, gamma family.
# `sampler_unif()` comes from `tests/testthat/helper.R`.

test_that("prim_acos", {
  verify_grad_uni(prim_acos, torch::torch_acos, tol = 1e-4, gen = sampler_unif(-0.95, 0.95))
})

test_that("prim_acosh", {
  verify_grad_uni(prim_acosh, torch::torch_acosh, tol = 1e-4, gen = sampler_unif(1.5, 5))
})

test_that("prim_asin", {
  verify_grad_uni(prim_asin, torch::torch_asin, tol = 1e-4, gen = sampler_unif(-0.95, 0.95))
})

test_that("prim_asinh", {
  verify_grad_uni(prim_asinh, torch::torch_asinh, tol = 1e-5)
})

test_that("prim_atan", {
  verify_grad_uni(prim_atan, torch::torch_atan, tol = 1e-5)
})

test_that("prim_atanh", {
  verify_grad_uni(prim_atanh, torch::torch_atanh, tol = 1e-4, gen = sampler_unif(-0.95, 0.95))
})

test_that("prim_cosh", {
  verify_grad_uni(prim_cosh, torch::torch_cosh, tol = 1e-4)
})

test_that("prim_sinh", {
  verify_grad_uni(prim_sinh, torch::torch_sinh, tol = 1e-4)
})

test_that("prim_digamma", {
  verify_grad_uni(prim_digamma, torch::torch_digamma, tol = 1e-4, gen = sampler_unif(0.5, 5))
})

test_that("prim_lgamma", {
  verify_grad_uni(prim_lgamma, torch::torch_lgamma, tol = 1e-4, gen = sampler_unif(0.5, 5))
})

test_that("prim_polygamma", {
  # Verify gradient w.r.t. x against torch::torch_polygamma
  shp <- c(2, 3)
  for (n_val in c(1L, 2L)) {
    x <- sampler_unif(0.5, 5)(shp, "f32")
    n_arr <- array(rep(n_val, prod(shp)), shp)

    f_nv <- function(x_t) {
      n_t <- prim_fill(as.numeric(n_val), dtype = dtype(x_t), shape = shape(x_t))
      out <- prim_polygamma(n_t, x_t)
      nv_reduce_sum(out, axes = seq_along(shape(out)), drop = TRUE)
    }
    grad_nv <- jit(gradient(f_nv))(nv_array(x, dtype = "f32"))[[1L]]

    x_th <- torch::torch_tensor(x, requires_grad = TRUE, dtype = torch::torch_float32())
    torch::torch_sum(torch::torch_polygamma(n_val, x_th))$backward()

    testthat::expect_equal(
      tengen::as_array(grad_nv),
      as_array_torch(x_th$grad),
      tolerance = 1e-4
    )
  }
})

test_that("prim_erf", {
  verify_grad_uni(prim_erf, torch::torch_erf, tol = 1e-4)
})

test_that("prim_erfc", {
  verify_grad_uni(prim_erfc, torch::torch_erfc, tol = 1e-4)
})

test_that("prim_erf_inv", {
  verify_grad_uni(prim_erf_inv, torch::torch_erfinv, tol = 1e-4, gen = sampler_unif(-0.95, 0.95))
})

test_that("prim_abs", {
  verify_grad_uni_tensor(
    prim_abs,
    torch::torch_abs,
    tol = 1e-5
  )
})

test_that("prim_max", {
  verify_grad_biv(prim_max, torch::torch_maximum, tol = 1e-5)
})

test_that("prim_min", {
  verify_grad_biv(prim_min, torch::torch_minimum, tol = 1e-5)
})

test_that("prim_floor", {
  verify_grad_uni(prim_floor, torch::torch_floor)
})

test_that("prim_ceil", {
  verify_grad_uni(prim_ceil, torch::torch_ceil)
})

test_that("prim_sign", {
  verify_grad_uni(prim_sign, torch::torch_sign)
})

test_that("prim_round", {
  verify_grad_uni(prim_round, torch::torch_round)
})

test_that("prim_cbrt", {
  verify_grad_uni(
    prim_cbrt,
    \(x) torch::torch_pow(x, 1 / 3),
    non_negative = TRUE,
    tol = 1e-4
  )
})

test_that("prim_expm1", {
  withr::local_seed(12)
  verify_grad_uni(prim_expm1, torch::torch_expm1, tol = 1e-5)
})

test_that("prim_log1p", {
  verify_grad_uni(prim_log1p, torch::torch_log1p, non_negative = TRUE, tol = 1e-5)
})

test_that("prim_logistic", {
  verify_grad_uni(prim_logistic, torch::torch_sigmoid, tol = 1e-5)
})

test_that("prim_clamp", {
  shp <- c(2L, 3L)
  dtype <- "f32"

  x_arr <- array(sample(c(-0.6, -0.5, -0.1, 0.0, 0.1, 0.5, 0.6), prod(shp), replace = TRUE), shp)
  x_nv <- nv_array(x_arr, dtype = dtype)
  x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())

  min_val <- -0.5
  max_val <- 0.5

  f_nv <- function(x) {
    y <- prim_clamp(min_val, x, max_val)
    nv_reduce_sum(y, axes = seq_len(naxes(y)), drop = TRUE)
  }

  grads_nv <- jit(gradient(f_nv))(x_nv)

  out_th <- torch::torch_clamp(x_th, min = min_val, max = max_val)
  torch::torch_sum(out_th)$backward()

  expect_equal(
    tengen::as_array(grads_nv[[1L]]),
    as_array_torch(x_th$grad),
    tolerance = 1e-5
  )
})

test_that("prim_reverse", {
  verify_grad_uni(
    prim_reverse,
    torch::torch_flip,
    naxes = 3L,
    args_f = \(shp, dtype) {
      axes_to_reverse <- sample(seq_along(shp), size = sample.int(length(shp), 1L))
      list(
        list(axes = axes_to_reverse),
        list(axes = axes_to_reverse)
      )
    },
    tol = 1e-5,
    skip_scalar = TRUE
  )
})

test_that("prim_atan2", {
  # Avoid (0, 0) which is undefined.
  verify_grad_biv(
    prim_atan2,
    torch::torch_atan2,
    tol = 1e-5,
    gen_lhs = sampler_nonzero(0.5),
    gen_rhs = sampler_nonzero(0.5)
  )
})

test_that("prim_concatenate", {
  verify_grad_concatenate <- function(shapes, axis = 2L, dtype = "f32", tol = 1e-5) {
    n <- length(shapes)
    arrs <- lapply(shapes, function(shp) generate_test_data(shp, dtype = dtype))
    nvs <- lapply(arrs, function(arr) nv_array(arr, dtype = dtype))
    ths <- lapply(arrs, function(arr) torch::torch_tensor(arr, requires_grad = TRUE))

    f_nv <- function(...) {
      args <- list(...)
      out <- do.call(prim_concatenate, c(args, list(axis = axis)))
      nv_reduce_sum(out, axes = seq_len(naxes(out)), drop = TRUE)
    }

    grads_nv <- do.call(jit(gradient(f_nv)), nvs)

    out_th <- torch::torch_cat(ths, dim = axis)
    torch::torch_sum(out_th)$backward()

    for (i in seq_len(n)) {
      testthat::expect_equal(
        tengen::as_array(grads_nv[[i]]),
        as_array_torch(ths[[i]]$grad),
        tolerance = tol
      )
    }
  }
  verify_grad_concatenate(list(c(2L, 3L), c(2L, 4L)))
  verify_grad_concatenate(list(c(2L, 2L), c(2L, 3L), c(2L, 1L)))
  verify_grad_concatenate(list(c(1, 3), c(2, 3)), 1L)
})

test_that("prim_reduce_prod", {
  # Avoid division by zero in the per-element gradient.
  shp <- c(2L, 3L)
  dtype <- "f32"

  check_against_torch <- function(x_arr, axis) {
    x_nv <- nv_array(x_arr, dtype = "f32")
    x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())

    f_nv <- function(x) {
      y <- prim_reduce_prod(x, axes = axis, drop = TRUE)
      nv_reduce_sum(y, axes = seq_along(shape(y)), drop = TRUE)
    }
    grads_nv <- jit(gradient(f_nv))(x_nv)

    out_th <- torch::torch_prod(x_th, dim = axis, keepdim = FALSE)
    torch::torch_sum(out_th)$backward()

    expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-4)
  }

  x <- array(runif(6), dim = c(2, 3))
  check_against_torch(x, 1)

  # Safe at zeros: matches PyTorch's prod_safe_zeros_backward.
  x_zero <- array(c(2, 0, 5, 1, 4, 6), dim = c(2L, 3L))
  check_against_torch(x_zero, 2L)

  x_two_zeros <- array(c(2, 0, 0, 1, 4, 6), dim = c(2L, 3L))
  check_against_torch(x_two_zeros, 2L)

  # Reducing multiple axes at once is not supported by torch::torch_prod,
  # so check against a hand-crafted expected gradient.
  x_multi <- array(c(1, 2, 3, 4, 5, 6), dim = c(2L, 3L))
  x_nv <- nv_array(x_multi, dtype = "f32")
  f_multi <- function(x) prim_reduce_prod(x, axes = c(1L, 2L), drop = TRUE)
  grads_nv <- jit(gradient(f_multi))(x_nv)
  expected <- array(prod(x_multi) / x_multi, dim = dim(x_multi))
  expect_equal(tengen::as_array(grads_nv[[1L]]), expected, tolerance = 1e-4)
})

describe("prim_static_slice", {
  verify_slice_grad <- function(shp, start_indices, limit_indices, strides, torch_slice_fn) {
    dtype <- "f32"
    x_arr <- generate_test_data(shp, dtype = dtype)
    x_nv <- nv_array(x_arr, dtype = dtype)
    x_th <- torch::torch_tensor(x_arr, requires_grad = TRUE, dtype = torch::torch_float32())

    f_nv <- function(x) {
      out <- prim_static_slice(x, start_indices, limit_indices, strides)
      nv_reduce_sum(out, axes = seq_len(naxes(out)), drop = TRUE)
    }

    grads_nv <- jit(gradient(f_nv))(x_nv)
    out_th <- torch_slice_fn(x_th)
    torch::torch_sum(out_th)$backward()

    testthat::expect_equal(tengen::as_array(grads_nv[[1L]]), as_array_torch(x_th$grad), tolerance = 1e-5)
  }

  it("works with unit strides", {
    verify_slice_grad(
      c(4L, 5L),
      c(2L, 2L),
      c(4L, 4L),
      c(1L, 1L),
      \(x) x[2:4, 2:4]
    )
  })

  it("works with non-unit strides", {
    verify_slice_grad(
      c(6L, 8L),
      c(1L, 1L),
      c(6L, 8L),
      c(2L, 2L),
      \(x) {
        x[c(1, 3, 5), c(1, 3, 5, 7)]
      }
    )
  })
})

test_that("prim_remainder", {
  # Compare against torch_fmod (truncating, sign-of-dividend) which matches
  # StableHLO `remainder` semantics. torch_remainder is flooring and would only
  # agree with us on same-sign inputs.
  gen_nonzero <- function(shp, dtype) {
    vals <- generate_test_data(shp, dtype = dtype)
    # Shift values away from zero to avoid division by zero
    if (length(shp) == 0L) {
      if (abs(vals) < 0.5) vals <- vals + sign(vals + 0.1) * 1
    } else {
      vals[abs(vals) < 0.5] <- vals[abs(vals) < 0.5] + sign(vals[abs(vals) < 0.5] + 0.1) * 1
    }
    if (length(shp) == 0L) vals else array(vals, shp)
  }

  verify_grad_biv(
    prim_remainder,
    torch::torch_fmod,
    tol = 1e-5,
    gen_rhs = sampler_nonzero(0.5) # avoid zero divisors
  )
})

gen_spd_matrix <- function(n) {
  R <- matrix(rnorm(n * n), n, n)
  A <- R %*% t(R) + diag(n)
  array(A, dim = c(n, n))
}

gen_tri_matrix <- function(n, lower, unit_diagonal) {
  M <- matrix(0, n, n)
  if (lower) {
    M[lower.tri(M, diag = TRUE)] <- rnorm(n * (n + 1) / 2)
  } else {
    M[upper.tri(M, diag = TRUE)] <- rnorm(n * (n + 1) / 2)
  }
  if (unit_diagonal) {
    diag(M) <- 1
  }
  M
}

describe("prim_chol", {
  verify_cholesky_grad <- function(lower) {
    n <- sample(2:4, 1L)
    A_r <- gen_spd_matrix(n)

    A_anvl <- nv_array(A_r, dtype = "f64")
    A_torch <- torch::torch_tensor(A_r, requires_grad = TRUE, dtype = torch::torch_float64())

    f_anvl <- function(A) {
      L <- prim_chol(A, lower = lower)
      nv_reduce_sum(L, axes = c(1L, 2L))
    }
    grad_anvl <- as_array(jit(gradient(f_anvl))(A_anvl)[[1L]])

    L_torch <- torch::linalg_cholesky(A_torch)
    if (!lower) {
      L_torch <- L_torch$t()
    }
    torch::torch_sum(L_torch)$backward()
    grad_torch <- as_array_torch(A_torch$grad)

    expect_equal(grad_anvl, grad_torch, tolerance = 1e-5)
  }

  it("lower = TRUE", verify_cholesky_grad(lower = TRUE))
  it("lower = FALSE", verify_cholesky_grad(lower = FALSE))
})

describe("prim_triangular_solve", {
  verify_triangular_solve_grad <- function(left_side, lower, transpose_a, unit_diagonal) {
    n <- sample(2:4, 1L)
    m <- sample(1:3, 1L)
    a_r <- gen_tri_matrix(n, lower, unit_diagonal)
    b_r <- if (left_side) array(rnorm(n * m), c(n, m)) else array(rnorm(m * n), c(m, n))

    a_anvl <- nv_array(a_r, dtype = "f64")
    b_anvl <- nv_array(b_r, dtype = "f64")

    a_torch <- torch::torch_tensor(a_r, requires_grad = TRUE, dtype = torch::torch_float64())
    b_torch <- torch::torch_tensor(b_r, requires_grad = TRUE, dtype = torch::torch_float64())

    f_anvl <- function(a, b) {
      x <- prim_triangular_solve(
        a,
        b,
        left_side = left_side,
        lower = lower,
        unit_diagonal = unit_diagonal,
        transpose_a = transpose_a
      )
      nv_reduce_sum(x, axes = c(1L, 2L))
    }
    grads_anvl <- jit(gradient(f_anvl))(a_anvl, b_anvl)

    is_upper <- if (transpose_a) lower else !lower
    a_effective <- if (transpose_a) a_torch$t() else a_torch
    x_torch <- torch::linalg_solve_triangular(
      a_effective,
      b_torch,
      upper = is_upper,
      left = left_side,
      unitriangular = unit_diagonal
    )
    torch::torch_sum(x_torch)$backward()

    expect_equal(as_array(grads_anvl[[1L]]), as_array_torch(a_torch$grad), tolerance = 1e-5)
    expect_equal(as_array(grads_anvl[[2L]]), as_array_torch(b_torch$grad), tolerance = 1e-5)
  }

  it(
    "left_side, lower, no transpose",
    verify_triangular_solve_grad(
      left_side = TRUE,
      lower = TRUE,
      transpose_a = FALSE,
      unit_diagonal = FALSE
    )
  )
  it(
    "left_side, lower, transpose",
    verify_triangular_solve_grad(
      left_side = TRUE,
      lower = TRUE,
      transpose_a = TRUE,
      unit_diagonal = FALSE
    )
  )
  it(
    "left_side, upper, no transpose",
    verify_triangular_solve_grad(
      left_side = TRUE,
      lower = FALSE,
      transpose_a = FALSE,
      unit_diagonal = FALSE
    )
  )
  it(
    "right_side, lower, no transpose",
    verify_triangular_solve_grad(
      left_side = FALSE,
      lower = TRUE,
      transpose_a = FALSE,
      unit_diagonal = FALSE
    )
  )
  it(
    "right_side, upper, transpose",
    verify_triangular_solve_grad(
      left_side = FALSE,
      lower = FALSE,
      transpose_a = TRUE,
      unit_diagonal = FALSE
    )
  )
  it(
    "left_side, lower, unit_diagonal",
    verify_triangular_solve_grad(
      left_side = TRUE,
      lower = TRUE,
      transpose_a = FALSE,
      unit_diagonal = TRUE
    )
  )

  # Verify the gradient zeros out non-triangular elements even when input is dense
  verify_triangular_solve_masking <- function(lower, unit_diagonal) {
    n <- 3L
    a_r <- matrix(seq_len(n * n), n, n) + 0
    diag(a_r) <- n + seq_len(n)
    b_r <- matrix(rnorm(n * 2L), n, 2L)

    a <- nv_array(a_r, dtype = "f64")
    b <- nv_array(b_r, dtype = "f64")

    f <- function(a, b) {
      x <- prim_triangular_solve(
        a,
        b,
        left_side = TRUE,
        lower = lower,
        unit_diagonal = unit_diagonal,
        transpose_a = FALSE
      )
      nv_reduce_sum(x, axes = c(1L, 2L))
    }
    grad_a <- as_array(jit(gradient(f))(a, b)[[1L]])

    if (lower) {
      expect_true(all(grad_a[upper.tri(grad_a)] == 0))
    } else {
      expect_true(all(grad_a[lower.tri(grad_a)] == 0))
    }
    if (unit_diagonal) {
      expect_true(all(diag(grad_a) == 0))
    }
  }

  it("masking: lower", verify_triangular_solve_masking(lower = TRUE, unit_diagonal = FALSE))
  it("masking: upper", verify_triangular_solve_masking(lower = FALSE, unit_diagonal = FALSE))
  it("masking: lower, unit_diagonal", verify_triangular_solve_masking(lower = TRUE, unit_diagonal = TRUE))
  it("masking: upper, unit_diagonal", verify_triangular_solve_masking(lower = FALSE, unit_diagonal = TRUE))
})
