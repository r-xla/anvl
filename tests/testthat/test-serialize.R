# the main tests are in {pjrt}, hence we keep them minimal here

test_that("nv_serialize and nv_unserialize work for single array", {
  x <- nv_matrix(rnorm(12), nrow = 3)
  lst <- list(x = x)
  raw_data <- nv_serialize(lst)
  expect_type(raw_data, "raw")
  reloaded <- nv_unserialize(raw_data)
  expect_equal(lst, reloaded)
})

test_that("nv_save and nv_read works for a single array", {
  x <- nv_matrix(rnorm(12), nrow = 3)
  lst <- list(x = x)
  tmp <- tempfile(fileext = ".safetensors")
  nv_save(lst, tmp)
  reloaded <- nv_read(tmp)
  expect_equal(lst, reloaded)
})

test_that("nv_serialize and nv_unserialize work for quickr backend", {
  skip_if_no_quickr()
  x <- nv_matrix(1:6, nrow = 2, dtype = "i32", backend = "quickr")
  lst <- list(x = x)
  raw_data <- nv_serialize(lst)
  expect_type(raw_data, "raw")
  reloaded <- nv_unserialize(raw_data, backend = "quickr")
  expect_equal(backend(reloaded$x), "quickr")
  expect_equal(as_array(reloaded$x), as_array(x))
  expect_equal(dtype(reloaded$x), dtype(x))
  expect_equal(shape(reloaded$x), shape(x))
})

test_that("nv_save and nv_read work for quickr backend", {
  skip_if_no_quickr()
  x <- nv_array(c(1.5, 2.5, 3.5), dtype = "f64", backend = "quickr")
  lst <- list(x = x)
  tmp <- tempfile(fileext = ".safetensors")
  nv_save(lst, tmp)
  reloaded <- nv_read(tmp, backend = "quickr")
  expect_equal(backend(reloaded$x), "quickr")
  expect_equal(as_array(reloaded$x), as_array(x))
  expect_equal(dtype(reloaded$x), dtype(x))
})

test_that("serialization round-trips scalars and typed arrays", {
  scalar_tensor <- nv_scalar(1.0)
  typed_tensor <- nv_array(1.0, dtype = "f32")

  lst <- list(
    scalar = scalar_tensor,
    typed = typed_tensor
  )

  # Test with nv_serialize/nv_unserialize
  raw_data <- nv_serialize(lst)
  reloaded <- nv_unserialize(raw_data)
  expect_equal(lst$scalar, reloaded$scalar)
  expect_equal(lst$typed, reloaded$typed)

  # Test with nv_save/nv_read
  tmp <- tempfile(fileext = ".safetensors")
  nv_save(lst, tmp)
  reloaded2 <- nv_read(tmp)
  expect_equal(lst$scalar, reloaded2$scalar)
  expect_equal(lst$typed, reloaded2$typed)
})
