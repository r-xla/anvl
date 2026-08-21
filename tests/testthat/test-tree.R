# The Rtree module (src/tree.h, src/tree.cpp, R/tree.R). These tests were
# moved back from pjrt together with the Rtree module, plus coverage
# for the natively-exposed structural operations.

test_that("(un)flatten lists", {
  # unnested
  x <- list(a = 1, 2)
  out <- flatten(x)
  expect_equal(out, list(1, 2))
  expect_equal(unflatten(build_tree(x), out), x)

  # nested depth 1
  x1 <- list(list(1), list(a = 2), 3)
  out1 <- flatten(x1)
  expect_equal(out1, list(1, 2, 3))
  expect_equal(unflatten(build_tree(x1), out1), x1)

  # nested depth 0
  x2 <- 1L
  out2 <- flatten(x2)
  expect_equal(out2, list(1L))
  expect_equal(unflatten(build_tree(x2), out2), x2)

  # a classed object or atomic vector is a single leaf (not recursed)
  expect_equal(flatten(1:5), list(1:5))
  expect_equal(flatten(list(matrix(1:4, 2, 2), "hi")), list(matrix(1:4, 2, 2), "hi"))
  clsd <- structure(list(a = 1), class = "some_class")
  expect_equal(flatten(list(clsd, 2)), list(clsd, 2))
})

test_that("tree ops reject non-RTree arguments with a clean error", {
  # A plain value or a foreign external pointer must not be reinterpreted as an
  # RTree* (which would segfault); it must raise an R error instead.
  foreign <- pjrt_scalar(1)
  for (bad in list(1L, "x", list(1, 2), NULL, foreign)) {
    expect_error(tree_size(bad), "RTree")
    expect_error(tree_root_kind(bad), "RTree")
  }
  # tree_concat only asserts a list on the R side; the element check is native.
  expect_error(tree_concat(list(build_tree(1), 2L)), "RTree")
})

test_that("unflatten rejects a leaf-count mismatch", {
  tree <- build_tree(list(1, 2))
  expect_error(unflatten(tree, list(10, 20, 30)), "2")
  expect_error(unflatten(tree, list(10)), "2")
  # exact count still round-trips
  expect_equal(unflatten(tree, list(10, 20)), list(10, 20))
})

test_that("NULL is an empty node (no leaves, round-trips)", {
  # NULL contributes no leaves to the flat list...
  expect_equal(flatten(NULL), list())
  expect_equal(flatten(list(a = 1, b = NULL, c = 2)), list(1, 2))

  # ...but is preserved in the tree and restored by unflatten.
  x <- list(a = 1, b = NULL, c = 2)
  tree <- build_tree(x)
  expect_equal(tree_size(tree), 2L)
  expect_equal(unflatten(tree, flatten(x)), x)

  # nested NULL
  y <- list(p = list(1, NULL), q = NULL)
  expect_equal(flatten(y), list(1))
  expect_equal(unflatten(build_tree(y), flatten(y)), y)

  # top-level NULL
  expect_equal(unflatten(build_tree(NULL), flatten(NULL)), NULL)
})

test_that("NULL position is captured structurally (distinct trees)", {
  # f(x, NULL) and f(NULL, x) must not collapse to the same structure,
  # otherwise a jit cache could not tell them apart.
  expect_false(tree_equal(
    build_tree(list(1, NULL)),
    build_tree(list(NULL, 1))
  ))
  # but two structurally identical NULL-bearing trees match
  expect_true(tree_equal(
    build_tree(list(a = 1, b = NULL)),
    build_tree(list(a = 1, b = NULL))
  ))
})

test_that("tree_equal distinguishes structure, names, and arity", {
  expect_true(tree_equal(build_tree(list(a = 1)), build_tree(list(a = 2))))
  expect_false(tree_equal(build_tree(list(a = 1)), build_tree(list(b = 1))))
  expect_false(tree_equal(build_tree(list(1)), build_tree(list(1, 2))))
  expect_false(tree_equal(build_tree(list(1)), build_tree(1)))
  expect_false(tree_equal(build_tree(list(1, 2)), build_tree(list(a = 1, b = 2))))
  expect_true(tree_equal(
    build_tree(list(a = 1, b = list(c = 2))),
    build_tree(list(a = 9, b = list(c = 8)))
  ))
  # an unnamed empty list differs from a named empty list (has_names is captured
  # even with zero children)
  expect_false(tree_equal(
    build_tree(list()),
    build_tree(structure(list(), names = character(0)))
  ))
})

test_that("NA list names are rejected (would corrupt to \"NA\" and poison tree_equal)", {
  x <- setNames(list(1, 2), c("a", NA))
  expect_error(build_tree(x), "NA")
  expect_error(flatten(x), "NA")
  expect_error(
    tree_concat(list(build_tree(1), build_tree(2)), names = c("a", NA)),
    "NA"
  )
  # a literal "NA" name is still fine and stays distinct
  ok <- build_tree(setNames(list(1), "NA"))
  expect_equal(tree_child_names(ok), "NA")
})

test_that("tree_hash is consistent with tree_equal", {
  # Structurally equal trees (equal under tree_equal, even with differing leaf
  # values) hash equally.
  a <- build_tree(list(a = 1, b = list(c = 2, d = NULL)))
  b <- build_tree(list(a = 9, b = list(c = 8, d = NULL)))
  expect_true(tree_equal(a, b))
  expect_identical(tree_hash(a), tree_hash(b))

  # A single string is returned, and hashing is deterministic.
  expect_type(tree_hash(a), "character")
  expect_length(tree_hash(a), 1L)
  expect_identical(tree_hash(a), tree_hash(a))
})

test_that("tree_hash distinguishes structure, names, arity, and NULL position", {
  h <- function(x) tree_hash(build_tree(x))
  # node kind
  expect_false(h(1) == h(list(1)))
  expect_false(h(NULL) == h(1))
  # child names
  expect_false(h(list(a = 1)) == h(list(b = 1)))
  # arity
  expect_false(h(list(1)) == h(list(1, 2)))
  # NULL position (f(x, NULL) vs f(NULL, x))
  expect_false(h(list(1, NULL)) == h(list(NULL, 1)))
  # has_names is captured even with zero children: list() vs a named empty list
  expect_false(h(list()) == h(structure(list(), names = character(0))))
})

test_that("tree_root_kind and tree_child_kinds", {
  expect_equal(tree_root_kind(build_tree(1)), "leaf")
  expect_equal(tree_root_kind(build_tree(list(1))), "list")
  expect_equal(tree_root_kind(build_tree(NULL)), "null")
  expect_equal(
    tree_child_kinds(build_tree(list(1, list(2), NULL))),
    c("leaf", "list", "null")
  )
  expect_error(tree_child_kinds(build_tree(1)), "list node")
})

test_that("tree_child_names and tree_child_sizes", {
  tree <- build_tree(list(a = 1, b = list(2, 3), c = NULL))
  expect_equal(tree_child_names(tree), c("a", "b", "c"))
  expect_null(tree_child_names(build_tree(list(1, 2))))
  expect_null(tree_child_names(build_tree(1)))
  expect_equal(tree_child_names(build_tree(list(1, b = 2))), c("", "b"))

  expect_equal(tree_child_sizes(tree), c(1L, 2L, 0L))
})

test_that("tree_leaf_mask marks all leaves under matching top-level names", {
  tree <- build_tree(list(a = 1, b = list(2, list(3)), c = NULL, d = 4))
  expect_equal(tree_leaf_mask(tree, "b"), c(FALSE, TRUE, TRUE, FALSE))
  expect_equal(tree_leaf_mask(tree, c("a", "d")), c(TRUE, FALSE, FALSE, TRUE))
  # empty names -> all-FALSE; unknown names ignored
  expect_equal(tree_leaf_mask(tree, character()), rep(FALSE, 4))
  expect_equal(tree_leaf_mask(tree, "zzz"), rep(FALSE, 4))
  # an unnamed tree marks nothing
  expect_equal(tree_leaf_mask(build_tree(list(1, 2)), "a"), c(FALSE, FALSE))
})

test_that("tree_filter_by_names keeps children in tree order and reindexes", {
  x <- list(a = 1, b = 2, c = 3)
  tree <- build_tree(x)
  sub <- tree_filter_by_names(tree, c("a", "c"))
  expect_equal(tree_size(sub), 2L)
  expect_equal(unflatten(sub, x[c("a", "c")]), x[c("a", "c")])

  # selection order does not matter; tree order wins
  sub2 <- tree_filter_by_names(tree, c("c", "a"))
  expect_true(tree_equal(sub, sub2))

  # keeping everything preserves the structure
  all_kept <- tree_filter_by_names(tree, c("a", "b", "c"))
  expect_true(tree_equal(all_kept, tree))

  # nested children reindex contiguously
  y <- list(a = list(1, 2), b = 3, c = list(d = 4))
  sub3 <- tree_filter_by_names(build_tree(y), c("a", "c"))
  expect_equal(tree_size(sub3), 3L)
  expect_equal(
    unflatten(sub3, list(10, 20, 30)),
    list(a = list(10, 20), c = list(d = 30))
  )

  expect_error(tree_filter_by_names(build_tree(list(1, 2)), "a"), "names")
})

test_that("tree_concat renumbers leaves contiguously across children", {
  value_tree <- build_tree(list(1, 2))
  grad_tree <- build_tree(list(g = 3))
  combined <- tree_concat(list(value_tree, grad_tree), names = c("value", "grad"))
  expect_equal(tree_size(combined), 3L)
  expect_equal(
    unflatten(combined, list(10, 20, 30)),
    list(value = list(10, 20), grad = list(g = 30))
  )

  # unnamed parent
  combined2 <- tree_concat(list(build_tree(1), build_tree(2)))
  expect_equal(unflatten(combined2, list("x", "y")), list("x", "y"))

  expect_error(tree_concat(list(build_tree(1)), names = c("a", "b")), "names")
})

test_that("tree_repr / format / print render R-idiomatic literals", {
  reprs <- vapply(
    list(
      build_tree(1),
      build_tree(list()),
      build_tree(list(1, 2)),
      build_tree(list(a = 1, b = 2)),
      build_tree(list(1, b = 2)),
      build_tree(list(a = list(b = 1, c = 2), d = 3)),
      build_tree(list(a = 1, b = NULL)),
      # named lists are tagged "list<named>" so the named/unnamed distinction is
      # visible even with no children or all-empty names
      build_tree(structure(list(), names = character(0))),
      build_tree(structure(list(1, 2), names = c("", "")))
    ),
    tree_repr,
    character(1)
  )
  expect_identical(
    reprs,
    c(
      "*",
      "list()",
      "list(*, *)",
      "list<named>(a = *, b = *)",
      "list<named>(*, b = *)",
      "list<named>(a = list<named>(b = *, c = *), d = *)",
      "list<named>(a = *, b = NULL)",
      "list<named>()",
      "list<named>(*, *)"
    )
  )
  expect_identical(format(build_tree(list(a = 1))), "list<named>(a = *)")
  expect_output(print(build_tree(list(a = 1))), "list<named>(a = *)", fixed = TRUE)
})

test_that("tree_path", {
  p <- function(x, i) tree_path(build_tree(x), i)
  expect_identical(p(list(x = 1), 1L), "x")
  expect_identical(p(list(l = list(a = 1, b = 2)), 1L), "l$a")
  expect_identical(p(list(l = list(a = 1, b = 2)), 2L), "l$b")
  expect_identical(p(list(l = list(1, 2)), 2L), "l[[2]]")
  expect_identical(p(list(l = list(1, b = 2)), 1L), "l[[1]]")
  expect_identical(p(list(l = list(1, b = 2)), 2L), "l$b")
  expect_identical(p(list(l = list(list(a = 1))), 1L), "l[[1]]$a")
  expect_identical(p(list(x = 1, y = 2), 2L), "y")
  expect_identical(p(list(pair = list(list(a = 1), list(b = 2))), 2L), "pair[[2]]$b")
  # a leaf at the root has no path
  expect_identical(p(1, 1L), "")
  # an out-of-range index on a root leaf errors instead of returning ""
  expect_error(p(1, 99L), "no leaf")
  expect_error(p(list(x = 1), 5L), "no leaf")
})

test_that("tree_diff locates the first divergence", {
  d <- function(a, b) tree_diff(build_tree(a), build_tree(b))
  expect_identical(
    d(1, list(1, 2)),
    list(prefix = "", a = "*", b = "list(*, *)")
  )
  expect_identical(
    d(list(a = 1, b = 2), list(p = 1, q = 2)),
    list(prefix = "", a = "list<named>(a = *, b = *)", b = "list<named>(p = *, q = *)")
  )
  expect_identical(
    d(list(1, 2), list(1, 2, 3)),
    list(prefix = "", a = "list(*, *)", b = "list(*, *, *)")
  )
  expect_identical(
    d(list(list(a = 1), list(a = 1)), list(list(a = 1), list(a = 1, b = 2))),
    list(prefix = "[[2]]", a = "list<named>(a = *)", b = "list<named>(a = *, b = *)")
  )
  expect_identical(
    d(list(pair = list(list(a = 1), 0)), list(pair = list(list(a = 1), list(c = 0)))),
    list(prefix = "pair[[2]]", a = "*", b = "list<named>(c = *)")
  )
  expect_null(d(list(a = 1, b = 2), list(a = 1, b = 2)))

  # named-ness is now visible in the repr, so tree_diff distinguishes an empty
  # named list from an unnamed one (both previously rendered "list()")
  expect_identical(
    d(structure(list(), names = character(0)), list()),
    list(prefix = "", a = "list<named>()", b = "list()")
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
    err <- tryCatch(
      map_tree(
        list(a = 1, b = list(c = 2)),
        \(x) if (x == 2) cli::cli_abort("boom") else x
      ),
      error = identity
    )
    expect_match(conditionMessage(err), "leaf at `b$c`", fixed = TRUE)
    expect_match(conditionMessage(err$parent), "boom")
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
    err <- tryCatch(
      pmap_tree(
        list(
          list(model = list(weights = list(a = 1), bias = 2)),
          list(model = list(weights = list(a = 1), bias = list(z = 9)))
        ),
        `+`
      ),
      error = identity
    )
    msg <- cli::ansi_strip(conditionMessage(err))
    expect_match(msg, "must have the same structure")
    expect_match(msg, "First mismatch at `model$bias`", fixed = TRUE)
    expect_match(msg, "list<named>(z = *)", fixed = TRUE)
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

  f_flat <- flatten_fun(f, build_tree(args))
  expect_s3_class(f_flat, "FlattenedFunction")
  out <- do.call(f_flat, flatten(args))
  out <- do.call(unflatten, out)
  expect_equal(args, out)
})

test_that("property: random nested structures round-trip", {
  withr::local_seed(42)

  random_leaf <- function() {
    switch(
      sample.int(4L, 1L),
      rnorm(sample.int(3L, 1L)),
      sample.int(100L, 1L),
      structure(list(v = 1), class = "some_leaf_class"),
      "s"
    )
  }
  random_tree <- function(depth) {
    if (depth <= 0L || runif(1) < 0.4) {
      return(random_leaf())
    }
    n <- sample.int(4L, 1L) - 1L # 0-3 children
    children <- lapply(seq_len(n), function(i) {
      if (runif(1) < 0.15) NULL else random_tree(depth - 1L)
    })
    if (runif(1) < 0.5 && n > 0L) {
      names(children) <- make.unique(sample(letters, n, replace = TRUE))
    }
    children
  }

  # R-side reference implementations of flatten and tree_leaf_mask
  ref_flatten <- function(x) {
    if (is.null(x)) {
      return(list())
    }
    if (is.list(x) && !is.object(x)) {
      out <- lapply(unname(x), ref_flatten)
      return(Reduce(c, out) %||% list())
    }
    list(x)
  }
  ref_child_size <- function(x) length(ref_flatten(x))

  for (rep in 1:50) {
    x <- random_tree(3L)
    flat <- flatten(x)
    tree <- build_tree(x)
    expect_identical(flat, ref_flatten(x))
    expect_identical(tree_size(tree), length(flat))
    expect_identical(unflatten(tree, flat), x)
    expect_true(tree_equal(tree, build_tree(x)))

    # every leaf's path resolves
    if (length(flat) > 0 && is.list(x) && !is.object(x)) {
      for (i in seq_along(flat)) {
        expect_type(tree_path(tree, i), "character")
      }
    }

    # tree_leaf_mask agrees with the reference expansion
    if (is.list(x) && !is.object(x) && length(x) > 0) {
      nms <- names(x) %||% rep("", length(x))
      marked <- sample(unique(nms[nzchar(nms)]), size = min(2L, sum(nzchar(nms))))
      ref_mask <- rep(nms %in% marked, times = vapply(x, ref_child_size, integer(1)))
      expect_identical(tree_leaf_mask(tree, marked), ref_mask)
    }
  }
})
