str_to_torch_dtype <- function(str) {
  switch(
    str,
    "bool" = torch::torch_bool(),
    "f32" = torch::torch_float32(),
    "f64" = torch::torch_float64(),
    "i8" = torch::torch_int8(),
    "i16" = torch::torch_int16(),
    "i32" = torch::torch_int32(),
    "i64" = torch::torch_int64(),
    "ui8" = torch::torch_uint8(),
    cli_abort(sprintf("Unsupported dtype: %s", str))
  )
}

as_array_torch <- function(x) {
  if (!length(dim(x))) {
    torch::as_array(x)
  } else if (length(dim(x)) == 1L) {
    array(torch::as_array(x), dim = length(x))
  } else {
    torch::as_array(x)
  }
}

generate_test_data <- function(dimension, dtype = "f64", non_negative = FALSE) {
  data <- if (dtype == "bool") {
    sample(c(TRUE, FALSE), size = prod(dimension), replace = TRUE)
  } else if (dtype %in% c("ui8", "ui16", "ui32", "ui64")) {
    sample(0:20, size = prod(dimension), replace = TRUE)
  } else if (dtype %in% c("i8", "i16", "i32", "i64")) {
    test_data <- as.integer(rgeom(prod(dimension), .5))
    if (!non_negative) {
      test_data <- as.integer((-1)^rbinom(prod(dimension), 1, .5) * test_data)
    }
    test_data
  } else {
    if (!non_negative) {
      rnorm(prod(dimension), mean = 0, sd = 1)
    } else {
      rchisq(prod(dimension), df = 1)
    }
  }

  # For scalars (dimension = integer()), return the value directly
  # For arrays, wrap in array() to preserve dimensions
  if (length(dimension) == 0L) data else array(data, dim = dimension)
}

make_nv <- function(x, dtype) {
  if (!length(dim(x))) nv_scalar(x, dtype = dtype) else nv_array(x, dtype = dtype)
}

make_torch <- function(x, dtype) {
  if (!length(dim(x))) {
    torch::torch_scalar_tensor(x, dtype = str_to_torch_dtype(dtype))
  } else {
    torch::torch_tensor(x, dtype = str_to_torch_dtype(dtype))
  }
}

expect_jit_torch_unary <- function(
  nv_fun,
  torch_fun,
  shp = integer(),
  dtype = "f32",
  args_list = list(),
  gen = NULL,
  non_negative = FALSE,
  tolerance = 1e-6
) {
  if (is.null(gen)) {
    vals <- generate_test_data(if (length(shp)) shp else integer(0), dtype = dtype, non_negative = non_negative)
    x <- if (length(shp)) array(vals, shp) else vals
  } else {
    x <- gen(shp, dtype)
  }
  x_nv <- make_nv(x, dtype)
  x_th <- make_torch(x, dtype)

  f <- jit(function(a, ...) do.call(nv_fun, c(list(a), list(...))))
  out_nv <- do.call(f, c(list(x_nv), args_list))
  out_th <- do.call(torch_fun, c(list(x_th), args_list))

  testthat::expect_equal(as_array(out_nv), as_array_torch(out_th), tolerance = tolerance)
}

expect_jit_torch_binary <- function(
  nv_fun,
  torch_fun,
  shp_x = integer(),
  shp_y = integer(),
  dtype = "f32",
  args_list = list(),
  gen_x = NULL,
  gen_y = NULL,
  non_negative = list(FALSE, FALSE)
) {
  if (length(non_negative) < 2) {
    non_negative <- rep(non_negative, 2)
  }
  if (is.null(gen_x)) {
    vals_x <- generate_test_data(
      if (length(shp_x)) shp_x else integer(0),
      dtype = dtype,
      non_negative = non_negative[[1]]
    )
    x <- if (length(shp_x)) array(vals_x, shp_x) else vals_x
  } else {
    x <- gen_x(shp_x, dtype)
  }
  if (is.null(gen_y)) {
    vals_y <- generate_test_data(
      if (length(shp_y)) shp_y else integer(0),
      dtype = dtype,
      non_negative = non_negative[[2]]
    )
    y <- if (length(shp_y)) array(vals_y, shp_y) else vals_y
  } else {
    y <- gen_y(shp_y, dtype)
  }
  x_nv <- make_nv(x, dtype)
  y_nv <- make_nv(y, dtype)
  x_th <- make_torch(x, dtype)
  y_th <- make_torch(y, dtype)

  f <- jit(function(a, b, ...) do.call(nv_fun, c(list(a, b), list(...))))
  out_nv <- do.call(f, c(list(x_nv, y_nv), args_list))
  out_th <- do.call(torch_fun, c(list(x_th, y_th), args_list))

  testthat::expect_equal(as_array(out_nv), as_array_torch(out_th), tolerance = 1e-6)
}
