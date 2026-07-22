test_that("jit: quickr backend compiles simple function", {
  skip_if_no_quickr()
  local_backend("quickr")

  f <- jit(function(x, y) x + y)

  expect_equal(as_array(f(nv_scalar(1L), nv_scalar(2L))), 3L)
})

test_that("jit: quickr backend returns AnvlArray", {
  skip_if_no_quickr()
  local_backend("quickr")

  f <- jit(function(x, y) x + y)
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(4, 5, 6))

  result <- f(x, y)
  expect_s3_class(result, "AnvlArray")
  expect_equal(backend(result), "quickr")
  expect_equal(as_array(result), array(c(5, 7, 9), dim = 3L))
})

test_that("jit: quickr backend preserves nested multi-output shapes and types", {
  skip_if_no_quickr()
  local_backend("quickr")

  f <- jit(
    function(x) {
      list(
        flags = x > 1L,
        payload = list(shifted = x + 1L)
      )
    }
  )

  out <- f(nv_array(1:3))

  expect_equal(dtype(out$flags), as_dtype("bool"))
  expect_equal(dtype(out$payload$shifted), as_dtype("i32"))
  expect_identical(shape(out$flags), 3L)
  expect_identical(shape(out$payload$shifted), 3L)
  expect_identical(as_array(out$flags), array(c(FALSE, TRUE, TRUE), dim = 3L))
  expect_identical(as_array(out$payload$shifted), array(2:4, dim = 3L))
})

test_that("jit: quickr backend does not support donate", {
  skip_if_no_quickr()
  local_backend("quickr")
  expect_error(
    jit(function(x) x, backend = "quickr", donate = "x"),
    "donate",
    fixed = TRUE
  )
})

test_that("jit: quickr backend supports unwrap = TRUE", {
  skip_if_no_quickr()
  local_backend("quickr")

  f <- jit(function(x, y) x + y, unwrap = TRUE)
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(4, 5, 6))

  result <- f(x, y)
  expect_false(inherits(result, "AnvlArray"))
  expect_equal(as.numeric(result), c(5, 7, 9))
})

test_that("jit: quickr backend unwrap = TRUE preserves nested output structure", {
  skip_if_no_quickr()
  local_backend("quickr")

  f <- jit(
    function(x) {
      list(
        flags = x > 1L,
        payload = list(shifted = x + 1L)
      )
    },
    unwrap = TRUE
  )

  out <- f(nv_array(1:3))

  expect_false(inherits(out$flags, "AnvlArray"))
  expect_false(inherits(out$payload$shifted, "AnvlArray"))
  expect_equal(as.logical(out$flags), c(FALSE, TRUE, TRUE))
  expect_equal(as.integer(out$payload$shifted), 2:4)
})

test_that("jit: quickr backend traces floating literals as f32", {
  graph <- trace_fn(
    function() 1.0,
    list(),
    desc = local_descriptor()
  )

  expect_equal(dtype(graph$outputs[[1L]]$aval), as_dtype("f32"))
})

test_that("graph_to_quickr_r_function lowers a graph to a plain R function", {
  skip_if_no_quickr()
  local_backend("quickr")

  graph <- trace_fn(
    function(x) x + 1L,
    list(x = nv_scalar(1.0, dtype = "f64")),
    desc = local_descriptor()
  )

  f <- graph_to_quickr_r_function(graph)

  expect_equal(f(2), 3)
})

test_that("quickr_device can be compared", {
  dev0 <- quickr_device("cpu")
  dev1 <- quickr_device("cpu")
  expect_true(dev0 == dev1)
})
