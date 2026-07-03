# The pytree module lives in pjrt (see pjrt's test-tree.R for its test suite);
# anvl re-exports the user-facing names.
test_that("pjrt tree API is re-exported and functional", {
  x <- list(a = 1, b = list(2), c = NULL)
  expect_equal(flatten(x), list(1, 2))
  tree <- build_tree(x)
  expect_equal(tree_size(tree), 2L)
  expect_equal(unflatten(tree, flatten(x)), x)
  expect_equal(tree_path(tree, 1L), "a")
  expect_equal(map_tree(list(a = 1), \(v) v + 1), list(a = 2))
  expect_equal(pmap_tree(list(list(1), list(2)), `+`), list(3))
})
