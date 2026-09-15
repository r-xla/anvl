test_that("nv_rng_state", {
  out1 <- nv_runif(
    initial_state = nv_rng_state(1L),
    dtype = "f64",
    shape = c(10, 5)
  )

  out2 <- nv_runif(
    initial_state = nv_rng_state(1L),
    dtype = "f64",
    shape = c(10, 5)
  )

  # check resulting states
  expect_equal(as_array(out1[[1]]), as_array(out2[[1]]))
  # check random variables
  expect_equal(as_array(out1[[2]]), as_array(out2[[2]]))
})
