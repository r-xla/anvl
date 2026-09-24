test_that("graph_runner_pjrt can run zero-input graphs", {
  testthat::skip_if_not_installed("pjrt")
  testthat::skip_if_not_installed("stablehlo")

  graph <- trace_fn(function() nv_fill(3.25, shape = integer(), dtype = "f64"), list())
  run <- graph_runner_pjrt(graph)
  expect_equal(run(), 3.25)
})
