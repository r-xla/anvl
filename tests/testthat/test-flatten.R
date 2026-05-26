test_that("(un)flatten lists", {
  # unnested
  x <- list(a = 1, 2)
  out <- flatten(x)
  expect_equal(out, list(1, 2))
  expect_equal(
    do.call(unflatten, list(build_tree(x), out)),
    x
  )

  # nested depth 1
  x1 <- list(list(1), list(a = 2), 3)
  out1 <- flatten(x1)
  expect_equal(out1, list(1, 2, 3))
  expect_equal(
    do.call(unflatten, list(build_tree(x1), out1)),
    x1
  )

  # nested depth 0
  x2 <- 1L
  out2 <- flatten(x2)
  expect_equal(out2, list(1L))
  expect_equal(
    do.call(unflatten, list(build_tree(x2), out2)),
    x2
  )
})

describe("map_tree", {
  it("preserves structure and applies f to leaves", {
    # leaf input
    expect_equal(map_tree(1, \(x) x + 1), 2)

    # flat list
    expect_equal(
      map_tree(list(a = 1, b = 2), \(x) x * 2),
      list(a = 2, b = 4)
    )

    # nested
    expect_equal(
      map_tree(list(a = 1, b = list(c = 2, d = 3)), \(x) x + 1),
      list(a = 2, b = list(c = 3, d = 4))
    )

    # extra args are forwarded to f
    expect_equal(
      map_tree(list(a = 1, b = list(c = 2)), `+`, 10),
      list(a = 11, b = list(c = 12))
    )
  })

  it("reports the leaf path on error", {
    expect_snapshot(
      map_tree(
        list(a = 1, b = list(c = 2)),
        \(x) if (x == 2) cli::cli_abort("boom") else x
      ),
      error = TRUE
    )
  })
})

describe("pmap_tree", {
  it("applies .f leaf-wise across multiple trees", {
    expect_equal(
      pmap_tree(list(list(a = 1, b = 2), list(a = 10, b = 20)), `+`),
      list(a = 11, b = 22)
    )
  })

  it("errors when trees have different structure", {
    expect_snapshot(
      pmap_tree(
        list(
          list(model = list(weights = list(a = 1), bias = 2)),
          list(model = list(weights = list(a = 1), bias = list(z = 9)))
        ),
        `+`
      ),
      error = TRUE
    )
  })
})

describe("tree node formatting", {
  it("renders LeafNode and ListNode as R-idiomatic literals", {
    expect_snapshot({
      # leaf at root
      build_tree(1)
      # empty list
      build_tree(list())
      # anonymous flat list
      build_tree(list(1, 2))
      # named flat list
      build_tree(list(a = 1, b = 2))
      # mixed named/unnamed
      build_tree(list(1, b = 2))
      # nested
      build_tree(list(a = list(b = 1, c = 2), d = 3))
    })
  })
})

describe("tree_diff", {
  it("locates the first divergence", {
    expect_snapshot({
      # leaf vs list at root
      tree_diff(build_tree(1), build_tree(list(1, 2)))
      # names mismatch at root (no shared prefix)
      tree_diff(build_tree(list(a = 1, b = 2)), build_tree(list(p = 1, q = 2)))
      # length mismatch in unnamed lists
      tree_diff(build_tree(list(1, 2)), build_tree(list(1, 2, 3)))
      # divergence inside a positional list
      tree_diff(
        build_tree(list(list(a = 1), list(a = 1))),
        build_tree(list(list(a = 1), list(a = 1, b = 2)))
      )
      # nested named-then-positional path
      tree_diff(
        build_tree(list(pair = list(list(a = 1), 0))),
        build_tree(list(pair = list(list(a = 1), list(c = 0))))
      )
      # identical trees -> NULL
      tree_diff(build_tree(list(a = 1, b = 2)), build_tree(list(a = 1, b = 2)))
    })
  })
})

test_that("tree_path", {
  expect_snapshot({
    tree_path(build_tree(list(x = 1)), 1L)
    tree_path(build_tree(list(l = list(a = 1, b = 2))), 1L)
    tree_path(build_tree(list(l = list(a = 1, b = 2))), 2L)
    tree_path(build_tree(list(l = list(1, 2))), 2L)
    tree_path(build_tree(list(l = list(1, b = 2))), 1L)
    tree_path(build_tree(list(l = list(1, b = 2))), 2L)
    tree_path(build_tree(list(l = list(list(a = 1)))), 1L)
    tree_path(build_tree(list(x = 1, y = 2)), 2L)
    tree_path(build_tree(list(pair = list(list(a = 1), list(b = 2)))), 2L)
  })
})

test_that("flatten_fun", {
  f <- function(a, b) {
    list(a, b)
  }
  args <- list(
    list(list(a = 1), list(2)),
    list(b = -1)
  )

  f_flat <- rlang::exec(flatten_fun, f, !!!args)
  expect_class(f_flat, "FlattenedFunction")
  out <- rlang::exec(f_flat, !!!flatten(args))
  args_flat <- flatten(args)
  out <- do.call(unflatten, out)
  expect_equal(args, out)
})
