#' @title Flatten
#' @description
#' Recursively flattens a nested list into a single flat list containing only
#' the leaf values, preserving left-to-right order.
#'
#' A bare (class-less) list is recursed into; `NULL` contributes no leaves;
#' everything else (a classed object, an atomic, a function, ...) is a leaf.
#'
#' Use [build_tree()] to capture the nesting structure so it can be restored
#' with [unflatten()].
#' @param x (any)\cr
#'   Object to flatten.
#' @return `list()` containing the flattened values.
#' @seealso [build_tree()], [unflatten()], [tree_size()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' flatten(x)
#'
#' flatten(list(1:3, "hello"))
#' @export
flatten <- function(x) {
  impl_tree_build_flatten(x)$leaves
}

#' @title Build Tree
#' @description
#' Captures the nesting structure of an object as an opaque `RTree`.
#' Each leaf in the input corresponds to its position in the flat list produced
#' by [flatten()]. The resulting tree can be passed to [unflatten()] to
#' reconstruct the original structure from a flat list.
#'
#' `NULL` is an empty node: it holds no leaf and contributes nothing to the
#' flat list, but is preserved in the tree (so `list(x, NULL)` and
#' `list(NULL, x)` have distinct trees).
#'
#' The `RTree` and its operations are pjrt's R analog of
#' \href{https://docs.jax.dev/en/latest/pytrees.html}{JAX's pytree}, which is
#' where the idea comes from.
#' @param x (any)\cr
#'   Object whose structure to capture. Bare lists are recursed into;
#'   everything else is a leaf.
#' @return A `RTree`.
#' @seealso [flatten()], [unflatten()], [tree_size()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' tree <- build_tree(x)
#' tree_size(tree)
#'
#' unflatten(tree, flatten(x))
#' @export
build_tree <- impl_tree_build

#' @title Unflatten
#' @description
#' Reconstructs a nested structure from a flat list by using a tree previously
#' created with [build_tree()].
#' @param tree (`RTree`)\cr
#'   Tree describing the target structure, as returned by [build_tree()].
#' @param x (`list`)\cr
#'   Flat list of leaf values, typically produced by [flatten()].
#' @return The reconstructed nested structure (list or single value).
#' @seealso [flatten()], [build_tree()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' tree <- build_tree(x)
#'
#' unflatten(tree, flatten(x))
#' unflatten(tree, list(10, 20, 30))
#' @export
unflatten <- impl_tree_unflatten

#' @title Tree Size
#' @description
#' Counts the number of leaf nodes in a tree. This equals the length of the
#' flat list produced by [flatten()] on the original structure.
#' @param tree (`RTree`)\cr
#'   A tree as returned by [build_tree()].
#' @return `integer(1)`
#' @seealso [build_tree()], [flatten()]
#' @examples
#' tree_size(build_tree(list(a = 1, b = list(c = 2, d = 3))))
#' @export
tree_size <- impl_tree_size

#' @title Compare Tree Structures
#' @description
#' Structural equality of two trees: identical node kinds, child structure,
#' leaf positions, and names.
#' @param a,b (`RTree`)\cr
#'   Trees to compare, as returned by [build_tree()].
#' @return `logical(1)`
#' @seealso [tree_diff()] to locate the first divergence.
#' @examples
#' tree_equal(build_tree(list(a = 1)), build_tree(list(a = 2)))
#' tree_equal(build_tree(list(a = 1)), build_tree(list(b = 1)))
#' @export
tree_equal <- impl_tree_equal

#' @title Structural Tree Hash
#' @description
#' A hash of a tree's structure -- node kinds, child structure, leaf positions,
#' and names -- consistent with [tree_equal()]: trees that are [tree_equal()]
#' hash to the same value. Intended as a cache key for tree-structured
#' dispatch.
#' @param tree (`RTree`)\cr
#'   A tree as returned by [build_tree()].
#' @return `character(1)`, the structural hash.
#' @seealso [tree_equal()], [build_tree()]
#' @examples
#' tree_hash(build_tree(list(a = 1))) == tree_hash(build_tree(list(a = 2)))
#' tree_hash(build_tree(list(a = 1))) == tree_hash(build_tree(list(b = 1)))
#' @export
tree_hash <- impl_tree_hash

#' @title Tree Root Node Kind
#' @description
#' The kind of a tree's root node: `"leaf"` for a single leaf, `"list"` for a
#' list node, or `"null"` for the empty (`NULL`) node.
#' @param tree (`RTree`)\cr
#'   A tree as returned by [build_tree()].
#' @return `character(1)`
#' @examples
#' tree_root_kind(build_tree(1))
#' tree_root_kind(build_tree(list(1)))
#' tree_root_kind(build_tree(NULL))
#' @export
tree_root_kind <- impl_tree_kind

#' @title Top-Level Child Names
#' @description
#' The names of a list tree's top-level children (with `""` for unnamed
#' slots), or `NULL` if the root is not a list node or carries no names.
#' @param tree (`RTree`)\cr
#'   A tree as returned by [build_tree()].
#' @return A `character` vector of child names, or `NULL` when the root is not
#'   a list node or is an *unnamed* list. An empty but named list
#'   (`structure(list(), names = character(0))`) returns `character(0)`, not
#'   `NULL`: the tree preserves the named/unnamed distinction (see
#'   [tree_equal()]).
#' @examples
#' tree_child_names(build_tree(list(a = 1, b = 2)))
#' tree_child_names(build_tree(list(1, 2)))
#' @export
tree_child_names <- impl_tree_child_names

#' @title Top-Level Child Kinds
#' @description
#' The node kind (`"leaf"`, `"list"`, or `"null"`) of each top-level child of
#' a list tree.
#' @param tree (`RTree`)\cr
#'   A tree whose root is a list node.
#' @return A `character` vector, one element per top-level child.
#' @examples
#' tree_child_kinds(build_tree(list(1, list(2), NULL)))
#' @export
tree_child_kinds <- impl_tree_child_kinds

#' @title Top-Level Child Sizes
#' @description
#' The number of leaves under each top-level child of a list tree.
#' @param tree (`RTree`)\cr
#'   A tree whose root is a list node.
#' @return An `integer` vector, one element per top-level child.
#' @examples
#' tree_child_sizes(build_tree(list(a = 1, b = list(2, 3), c = NULL)))
#' @export
tree_child_sizes <- impl_tree_child_sizes

#' @title Tree Path
#' @description
#' Returns the human-readable path for a single leaf identified by its flat
#' index (e.g. `"a$b"` or `"[[2]]"`). Only descends into the branch containing
#' the target leaf, making it efficient for error reporting.
#' @param tree (`RTree`)\cr
#'   A tree as returned by [build_tree()].
#' @param i (`integer(1)`)\cr
#'   The flat index of the leaf.
#' @return `character(1)` (`""` for a leaf at the root).
#' @seealso [build_tree()], [flatten()]
#' @examples
#' tree_path(build_tree(list(a = 1, b = list(c = 2))), 2L)
#' @export
tree_path <- function(tree, i) {
  impl_tree_path(tree, as.integer(i))
}

#' @title Filter a Tree by Top-Level Names
#' @description
#' Keeps only the top-level children whose names match `names` (in tree
#' order), reindexing the remaining leaves so they map to contiguous positions
#' in a flat list.
#' @param tree (`RTree`)\cr
#'   A named list tree as returned by [build_tree()].
#' @param names (`character()`)\cr
#'   Names of children to keep.
#' @return A `RTree` containing only the selected children.
#' @seealso [build_tree()], [unflatten()]
#' @examples
#' x <- list(a = 1, b = 2, c = 3)
#' sub <- tree_filter_by_names(build_tree(x), c("a", "c"))
#' tree_size(sub)
#' unflatten(sub, x[c("a", "c")])
#' @export
tree_filter_by_names <- function(tree, names) {
  impl_tree_filter_by_names(tree, as.character(names))
}

#' @title Concatenate Trees Under a New Root
#' @description
#' Builds a new list tree whose children are the given trees, reassigning leaf
#' indices contiguously across all children (in order).
#' @param children (`list` of `RTree`)\cr
#'   The child trees.
#' @param names (`NULL` | `character()`)\cr
#'   Optional names for the children.
#' @return A `RTree`.
#' @examples
#' combined <- tree_concat(
#'   list(build_tree(list(1, 2)), build_tree(3)),
#'   names = c("value", "grad")
#' )
#' tree_size(combined)
#' @export
tree_concat <- function(children, names = NULL) {
  checkmate::assert_list(children)
  impl_tree_concat(children, names)
}

#' @title Per-Leaf Mask From Top-Level Groups
#' @description
#' A logical mask over the leaves (in flat order) that is `TRUE` for every
#' leaf sitting under a top-level child whose name is in `groups`. An empty
#' `groups` yields an all-`FALSE` mask.
#' @param tree (`RTree`)\cr
#'   A tree whose root is a list node.
#' @param groups (`character()`)\cr
#'   Top-level child names to mark.
#' @return A `logical` vector of length `tree_size(tree)`.
#' @seealso [tree_child_names()]
#' @examples
#' tree_leaf_mask(build_tree(list(a = 1, b = list(2, 3))), "b")
#' @export
tree_leaf_mask <- function(tree, groups) {
  impl_tree_mask_from_names(tree, as.character(groups))
}

#' @title Canonical Tree Representation
#' @description
#' A canonical structural string for a tree: `"*"` for a leaf, `"NULL"` for an
#' empty node, and `"list(...)"` for list nodes.
#' @param tree (`RTree`)\cr
#'   A tree as returned by [build_tree()].
#' @return `character(1)`
#' @examples
#' tree_repr(build_tree(list(a = 1, b = list(2, NULL))))
#' @export
tree_repr <- impl_tree_repr

#' @title Difference Between Trees
#' @description
#' Walks two trees in parallel and returns the first path/subtree pair where
#' they diverge, or `NULL` if they are structurally identical. The returned
#' `prefix` follows [tree_path()] syntax; `a` and `b` are the [tree_repr()]
#' strings of the diverging subtrees.
#' @param a,b (`RTree`)\cr
#'   Trees to compare, as returned by [build_tree()].
#' @return `NULL` if `a` and `b` are structurally identical, otherwise a list
#'   with elements `prefix`, `a`, and `b`.
#' @seealso [build_tree()], [tree_path()], [tree_equal()]
#' @examples
#' tree_diff(build_tree(list(a = 1)), build_tree(list(a = list(1, 2))))
#' @export
tree_diff <- impl_tree_diff

#' @title Map Over a Tree
#' @description
#' Apply a function to each leaf of a (possibly nested) list, preserving the
#' tree structure. Equivalent to flattening `.x` with [flatten()], applying
#' `.f` to each leaf, and reassembling with [unflatten()].
#' @param .x (any)\cr
#'   A leaf or a (nested) list of leaves.
#' @param .f (`function`)\cr
#'   Function to apply to each leaf of `.x`.
#' @param ... Additional arguments passed to `.f`.
#' @return An object with the same nesting structure as `.x`, where each leaf
#'   is the result of `.f(leaf, ...)`.
#' @seealso [flatten()], [build_tree()], [unflatten()]
#' @examples
#' map_tree(list(a = 1, b = list(c = 2, d = 3)), \(x) x + 1)
#' @export
map_tree <- function(.x, .f, ...) {
  built <- impl_tree_build_flatten(.x)
  tree <- built$tree
  flat <- built$leaves
  result <- lapply(seq_along(flat), function(i) {
    tryCatch(
      .f(flat[[i]], ...),
      error = function(e) {
        path <- tree_path(tree, i)
        loc <- if (nzchar(path)) path else "<root>"
        cli::cli_abort(
          "Error applying {.arg .f} to leaf at {.code {loc}}.",
          parent = e,
          call = NULL
        )
      }
    )
  })
  unflatten(tree, result)
}

#' @title Map Over Multiple Trees
#' @description
#' Apply a function leaf-wise over several trees with the same structure.
#' All trees in `.l` must have identical structure.
#' @param .l (`list`)\cr
#'   A non-empty list of trees, all with the same structure.
#' @param .f (`function`)\cr
#'   Function to call with one leaf from each tree (positional arguments, in
#'   the order given by `.l`).
#' @param ... Additional arguments passed to `.f` after the per-tree leaves.
#' @return A tree with the same structure as `.l[[1]]`, where each leaf is
#'   `.f(leaf_1, leaf_2, ..., leaf_n, ...)`.
#' @seealso [map_tree()], [flatten()], [unflatten()]
#' @examples
#' pmap_tree(list(list(a = 1, b = 2), list(a = 10, b = 20)), `+`)
#' @export
pmap_tree <- function(.l, .f, ...) {
  if (!is.list(.l) || length(.l) == 0L) {
    cli::cli_abort("{.arg .l} must be a non-empty list of trees.")
  }
  built <- impl_tree_build_flatten(.l[[1L]])
  tree <- built$tree
  flats <- vector("list", length(.l))
  flats[[1L]] <- built$leaves
  for (i in seq_along(.l)[-1L]) {
    other <- impl_tree_build_flatten(.l[[i]])
    if (!tree_equal(tree, other$tree)) {
      diff <- tree_diff(tree, other$tree)
      header <- if (nzchar(diff$prefix)) {
        "First mismatch at {.code {diff$prefix}}:"
      } else {
        "Trees differ at the root:"
      }
      cli::cli_abort(c(
        "All trees in {.arg .l} must have the same structure.",
        "x" = header,
        "*" = "{.code .l[[1]]}: {diff$a}",
        "*" = "{.code .l[[{i}]]}: {diff$b}"
      ))
    }
    flats[[i]] <- other$leaves
  }
  n <- length(flats[[1L]])
  result <- lapply(seq_len(n), function(i) {
    tryCatch(
      do.call(.f, c(lapply(flats, `[[`, i), list(...))),
      error = function(e) {
        path <- tree_path(tree, i)
        loc <- if (nzchar(path)) path else "<root>"
        cli::cli_abort(
          "Error applying {.arg .f} to leaves at {.code {loc}}.",
          parent = e,
          call = NULL
        )
      }
    )
  })
  unflatten(tree, result)
}

#' @title Flatten a Function
#' @description
#' Flattens a function taking in possibly nested `RTree` objects so it takes in flat inputs
#' and return flat outputs (along the returned tree structure).
#' @param f (`function`)\cr
#'   Function to wrap.
#' @param in_tree (`RTree`)\cr
#'   Tree describing the (possibly nested) argument list of `f`, i.e.
#'   `build_tree(list(arg1, arg2, ...))`. The flat inputs to the wrapper follow
#'   this structure.
#' @return A `FlattenedFunction`.
#' @seealso [build_tree()], [flatten()], [unflatten()]
#' @examples
#' f <- function(x) list(sum = x$a + x$b)
#' ff <- flatten_fun(f, build_tree(list(list(a = 1, b = 2))))
#' ff(3, 4)
#' @export
flatten_fun <- function(f, in_tree) {
  f_orig <- f
  f <- function(...) {
    args <- unflatten(in_tree, list(...))
    outs <- do.call(f_orig, args)
    list(
      build_tree(outs),
      flatten(outs)
    )
  }
  class(f) <- "FlattenedFunction"
  f
}

#' @export
format.RTree <- function(x, ...) {
  tree_repr(x)
}

#' @export
print.RTree <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}
