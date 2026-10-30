# the main tests are in {pjrt}, hence we keep them minimal here

test_that("nv_serialize and nv_unserialize work for single array", {
  x <- nv_matrix(rnorm(12), nrow = 3)
  lst <- list(x = x)
  # A bare array, not a list: `assert_list(types = )` subsets its input and an
  # `AnvlArray` has a `[` method, so the assertion itself went into
  # `nv_subset()` and reported a subsetting error.
  expect_error(nv_serialize(x), "must be a named list of arrays")
  raw_data <- nv_serialize(lst)
  expect_type(raw_data, "raw")
  reloaded <- nv_unserialize(raw_data)
  expect_equal(lst, reloaded)
})

test_that("nv_save and nv_read works for a single array", {
  x <- nv_matrix(rnorm(12), nrow = 3)
  lst <- list(x = x)
  expect_error(nv_save(x, tempfile()), "must be a named list of arrays")
  tmp <- tempfile(fileext = ".safetensors")
  # Writing to a connection is done for the side effect, so it returns
  # invisibly, as `nv_save()` does.
  con <- file(tempfile(), "wb")
  res <- withVisible(nv_serialize(lst, con))
  close(con)
  expect_null(res$value)
  expect_false(res$visible)
  nv_save(lst, tmp)
  reloaded <- nv_read(tmp)
  expect_equal(lst, reloaded)
})

# Arrays of every data type the r backend holds, a scalar, and a rank-3 array,
# whose bytes are written in row-major order.
serialize_fixture <- function() {
  list(
    f64 = nv_array(array(as.numeric(1:24) / 4, dim = c(2L, 3L, 4L)), dtype = "f64"),
    i32 = nv_matrix(-3:2, nrow = 2L, dtype = "i32"),
    bool = nv_array(c(TRUE, FALSE, TRUE)),
    scalar = nv_scalar(-2.5, dtype = "f64")
  )
}

expect_arrays_like <- function(actual, expected, backend) {
  expect_named(actual, names(expected))
  for (name in names(expected)) {
    expect_equal(backend(actual[[name]]), backend)
    expect_dtype(actual[[name]], dtype(expected[[name]]))
    expect_shape(actual[[name]], shape(expected[[name]]))
    expect_equal(as.vector(as_array(actual[[name]])), as.vector(as_array(expected[[name]])))
  }
}

test_that("nv_serialize and nv_unserialize work on the r backend", {
  local_backend("r")
  arrays <- serialize_fixture()
  raw_data <- nv_serialize(arrays)
  expect_type(raw_data, "raw")
  expect_equal(nv_unserialize(raw_data), arrays)
})

test_that("nv_save and nv_read work on the r backend", {
  local_backend("r")
  arrays <- serialize_fixture()
  tmp <- tempfile(fileext = ".safetensors")
  nv_save(arrays, tmp)
  expect_equal(nv_read(tmp), arrays)
})

test_that("arrays saved on one backend are read on another", {
  on_pjrt <- with_backend("pjrt", serialize_fixture())
  on_r <- with_backend("r", serialize_fixture())
  expect_arrays_like(with_backend("r", nv_unserialize(nv_serialize(on_pjrt))), on_pjrt, "r")
  expect_arrays_like(with_backend("pjrt", nv_unserialize(nv_serialize(on_r))), on_r, "pjrt")
  # The two backends write the same bytes.
  expect_identical(nv_serialize(on_r), nv_serialize(on_pjrt))
})

test_that("a data type the r backend cannot hold is an error when read on it", {
  raw_data <- with_backend("pjrt", nv_serialize(list(x = nv_array(1, dtype = "f32"))))
  expect_error(with_backend("r", nv_unserialize(raw_data)), "f32")
})

test_that("serialization round-trips scalars and typed arrays", {
  scalar_arr <- nv_scalar(1.0)
  typed_arr <- nv_array(1.0, dtype = "f32")

  lst <- list(
    scalar = scalar_arr,
    typed = typed_arr
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

test_that("nv_unserialize places the arrays on the default device", {
  local_default_device("cpu:1")
  lst <- list(x = nv_array(1:3))
  expect_equal(device(nv_unserialize(nv_serialize(lst))$x), nv_device("cpu:1"))
})
