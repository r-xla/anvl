# Pack and unpack
flatten_fun <- function(f, ..., in_node = NULL) {
  if (is.null(in_node)) {
    in_node <- build_tree(list(...))
  } else if (...length()) {
    cli_abort("in_node is not compatible with ... arguments")
  }
  f_orig <- f
  f <- function(...) {
    # We could do this out of the function and re-use,
    # but because we always jit, we don't worry about it
    # (at least for now)
    args <- unflatten(in_node, list(...))

    outs <- do.call(f_orig, args)
    list(
      build_tree(outs),
      flatten(outs)
    )
  }
  class(f) <- "FlattenedFunction"
  f
}

new_counter <- function() {
  y <- hashtab(size = 1L)
  y[["i"]] <- 0L
  y
}

#' @title Flatten
#' @description
#' Recursively flattens a nested list into a single flat list containing only the
#' leaf values, preserving left-to-right order.
#'
#' Currently only lists are flattened and all other objects are treated as leaves.
#'
#' Use [build_tree()] to capture the nesting structure so it can be restored with
#' [unflatten()].
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
  # NULL is an empty node: it contributes no leaves (see build_tree()).
  if (is.null(x)) {
    return(list())
  }
  UseMethod("flatten")
}

#' @export
flatten.list <- function(x) {
  out <- lapply(unname(x), flatten)
  Reduce(c, out)
}

#' @export
flatten.default <- function(x) {
  list(x)
}

#' @title Build Tree
#' @description
#' Captures the nesting structure of an object as a tree of `Node`s. Each leaf
#' in the input becomes a `LeafNode` with an integer index corresponding to its
#' position in the flat list produced by [flatten()]. Lists become `ListNode`s
#' that record child nodes and names. The resulting tree can be passed to
#' [unflatten()] to reconstruct the original structure from a flat list.
#' @param x (any)\cr
#'   Object whose structure to capture. Lists are recursed into; everything else
#'   is a leaf.
#' @param counter (NULL | environment)\cr
#'   Internal counter for assigning leaf indices. Mostly used internally and otherwise left as
#'   `NULL` (default.)
#' @return A `Node` (`LeafNode` for scalars, `ListNode` for lists).
#' @seealso [flatten()], [unflatten()], [tree_size()], [reindex_tree()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' tree <- build_tree(x)
#' tree_size(tree)
#'
#' flat <- flatten(x)
#' unflatten(tree, flat)
#' @export
build_tree <- function(x, counter = NULL) {
  # NULL becomes an empty node: no leaf index is allocated (the counter is
  # left untouched), so it contributes nothing to the flat list while still
  # recording its position in the tree.
  if (is.null(x)) {
    return(NullNode())
  }
  UseMethod("build_tree")
}

#' @export
build_tree.list <- function(x, counter = NULL) {
  counter <- counter %||% new_counter()
  out <- lapply(unname(x), build_tree, counter = counter)
  # basically non-recursive unlist that maintains list structure even for atomics (unlist(list(1, 2))
  # would become c(1, 2))
  ListNode(
    out,
    names(x)
  )
}

#' @export
build_tree.default <- function(x, counter = NULL) {
  counter <- counter %||% new_counter()
  i <- counter[["i"]] + 1L
  counter[["i"]] <- i
  LeafNode(i)
}

mark_some <- function(x, marked) {
  stopifnot(is.list(x))
  structure(list(data = x, marked = marked), class = "MarkedArgs")
}

#' @export
build_tree.MarkedArgs <- function(x, counter = NULL) {
  if (is.null(counter)) {
    counter <- new_counter()
  }

  n <- length(x$data)
  subsize <- vector("integer", n)
  nodes <- vector("list", n)
  prev <- 0L
  for (i in seq_along(x$data)) {
    nodes[[i]] <- build_tree(x$data[[i]], counter = counter)
    subsize[i] <- counter[["i"]] - prev
    prev <- prev + subsize[i]
  }

  if (!is.null(x$marked)) {
    is_marked <- rlang::names2(x$data) %in% x$marked
    is_marked_flat <- rep(is_marked, times = subsize)
  } else {
    cli_abort("Internal error: marked must be non-NULL")
  }

  MarkedListNode(nodes, names(x$data), is_marked_flat)
}


#' @title Unflatten
#' @description
#' Reconstructs a nested structure from a flat list by using a tree previously
#' created with [build_tree()]. Each `LeafNode` in the tree selects the
#' corresponding element from `x` by index, and `ListNode`s restore the
#' original nesting and names.
#' @param node (`Node`)\cr
#'   Tree describing the target structure, as returned by [build_tree()].
#' @param x (list)\cr
#'   Flat list of leaf values, typically produced by [flatten()].
#' @return The reconstructed nested structure (list or single value).
#' @seealso [flatten()], [build_tree()]
#' @examples
#' x <- list(a = 1, b = list(c = 2, d = 3))
#' tree <- build_tree(x)
#' flat <- flatten(x)
#'
#' unflatten(tree, flat)
#'
#' unflatten(tree, list(10, 20, 30))
#' @export
unflatten <- function(node, x) {
  # An empty node restores NULL without consuming a flat element.
  if (inherits(node, "NullNode")) {
    return(NULL)
  }
  UseMethod("unflatten")
}

#' @export
unflatten.LeafNode <- function(node, x) {
  x[[node$i]]
}

#' @export
unflatten.ListNode <- function(node, x) {
  stats::setNames(lapply(node$nodes, unflatten, x = x), node$names)
}

LeafNode <- function(i) {
  structure(list(i = i), class = c("LeafNode", "Node"))
}

# An empty node, used to represent NULL: it holds no leaf and contributes
# nothing to the flat list, but is preserved in the tree (and therefore in the
# jit cache key) so NULL arguments round-trip through unflatten(). Mirrors how
# JAX treats `None` as an empty pytree node.
NullNode <- function() {
  structure(list(), class = c("NullNode", "Node"))
}

ListNode <- function(nodes, names) {
  structure(
    list(
      nodes = unname(nodes),
      names = names
    ),
    class = c("ListNode", "Node")
  )
}

MarkedListNode <- function(nodes, names, marked) {
  structure(
    list(
      nodes = unname(nodes),
      names = names,
      marked = marked
    ),
    class = c("MarkedListNode", "ListNode", "Node")
  )
}

#' @export
format.LeafNode <- function(x, ...) "*"

#' @export
format.ListNode <- function(x, ...) {
  if (length(x$nodes) == 0L) {
    return("list()")
  }
  child_strs <- vapply(
    x$nodes,
    function(n) if (inherits(n, "NullNode")) "NULL" else format(n, ...),
    character(1L)
  )
  parts <- if (is.null(x$names)) {
    child_strs
  } else {
    has_name <- nzchar(x$names)
    ifelse(has_name, paste0(x$names, " = ", child_strs), child_strs)
  }
  paste0("list(", paste(parts, collapse = ", "), ")")
}

#' @export
print.LeafNode <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}

#' @export
print.ListNode <- function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}

#' @title Tree Size
#' @description
#' Counts the number of leaf nodes in a tree. This equals the length of the
#' flat list produced by [flatten()] on the original structure.
#' @param x (`Node`)\cr
#'   A tree node as returned by [build_tree()].
#' @return A scalar `integer`.
#' @seealso [build_tree()], [flatten()]
#' @examples
#' tree <- build_tree(list(a = 1, b = list(c = 2, d = 3)))
#' tree_size(tree)
#'
#' tree_size(build_tree(list(1)))
#' @export
tree_size <- function(x) {
  if (inherits(x, "NullNode")) {
    return(0L)
  }
  UseMethod("tree_size")
}

#' @export
tree_size.LeafNode <- function(x) {
  1L
}

#' @export
tree_size.ListNode <- function(x) {
  sum(vapply(x$nodes, tree_size, integer(1L)))
}

#' @export
tree_size.MarkedListNode <- function(x) {
  sum(vapply(x$nodes, tree_size, integer(1L)))
}

#' @title Tree Path
#' @description
#' Returns the human-readable path for a single leaf node identified by its
#' flat index. Only descends into the branch containing the target leaf,
#' making it efficient for error reporting.
#' @param node (`Node`)\cr
#'   A tree node as returned by [build_tree()].
#' @param i (`integer(1)`)\cr
#'   The flat index of the leaf (as stored in `LeafNode$i`).
#' @param prefix (`character(1)`)\cr
#'   Path prefix. Used internally during recursion; callers should leave as `""`.
#' @return A scalar `character` string.
#' @seealso [build_tree()], [flatten()]
#' @export
tree_path <- function(node, i, prefix = "") {
  # An empty node holds no leaf, so it can never be the target.
  if (inherits(node, "NullNode")) {
    return(NULL)
  }
  UseMethod("tree_path")
}

#' @export
tree_path.LeafNode <- function(node, i, prefix = "") {
  prefix
}

#' @export
tree_path.ListNode <- function(node, i, prefix = "") {
  for (j in seq_along(node$nodes)) {
    child <- node$nodes[[j]]
    nm <- if (!is.null(node$names)) node$names[j] else ""
    suffix <- if (nzchar(nm)) {
      if (nzchar(prefix)) paste0("$", nm) else nm
    } else {
      paste0("[[", j, "]]")
    }
    child_prefix <- paste0(prefix, suffix)
    if (inherits(child, "LeafNode")) {
      if (child$i == i) return(child_prefix)
    } else {
      result <- tree_path(child, i, child_prefix)
      if (!is.null(result)) return(result)
    }
  }
  NULL
}

#' @title Filter List Node
#' @description
#' Subsets a `ListNode` to keep only the children whose names match `names`,
#' then reindexes the leaf nodes so they map to contiguous positions in a flat
#' list. If all names are kept the original tree is returned unchanged.
#' @param tree (`ListNode`)\cr
#'   A named list node as returned by [build_tree()].
#' @param names (character)\cr
#'   Names of children to keep.
#' @return A `ListNode` containing only the selected children with reindexed
#'   leaves.
#' @seealso [build_tree()], [reindex_tree()], [unflatten()]
#' @examples
#' x <- list(a = 1, b = 2, c = 3)
#' tree <- build_tree(x)
#' sub <- filter_list_node(tree, c("a", "c"))
#' tree_size(sub)
#'
#' unflatten(sub, x[c("a", "c")])
#' @export
filter_list_node <- function(tree, names) {
  stopifnot(inherits(tree, "ListNode"))
  if (is.null(tree$names)) {
    cli_abort("tree must have names")
  }
  keep_idx <- which(tree$names %in% names)
  if (length(keep_idx) == length(tree$names)) {
    return(tree)
  }
  counter <- new_counter()
  renumbered_nodes <- lapply(tree$nodes[keep_idx], reindex_tree, counter = counter)
  ListNode(renumbered_nodes, tree$names[keep_idx])
}

#' @title Reindex Tree
#' @description
#' Reassigns leaf indices so they form a contiguous sequence starting from the
#' current counter value. This is used internally after filtering nodes from a tree (e.g.
#' via [filter_list_node()]) to ensure leaf indices still map correctly to
#' positions in a flat list.
#' Not intended for direct use.
#' @param x (`Node`)\cr
#'   A tree node to reindex.
#' @param counter (environment)\cr
#'   A mutable counter created by `new_counter()`.
#' @return A new `Node` with updated leaf indices.
#' @export
reindex_tree <- function(x, counter) {
  # An empty node has no leaf index to reassign.
  if (inherits(x, "NullNode")) {
    return(NullNode())
  }
  UseMethod("reindex_tree")
}

#' @export
reindex_tree.LeafNode <- function(x, counter) {
  i <- counter[["i"]] + 1L
  counter[["i"]] <- i
  LeafNode(i)
}

#' @export
reindex_tree.ListNode <- function(x, counter) {
  reindexed <- lapply(x$nodes, reindex_tree, counter = counter)
  ListNode(reindexed, x$names)
}

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
#' @seealso [flatten()], [build_tree()], [unflatten()], [await()]
#' @examples
#' map_tree(list(a = 1, b = list(c = 2, d = 3)), \(x) x + 1)
#' @export
map_tree <- function(.x, .f, ...) {
  tree <- build_tree(.x)
  flat <- flatten(.x)
  result <- lapply(seq_along(flat), function(i) {
    tryCatch(
      .f(flat[[i]], ...),
      error = function(e) {
        path <- tree_path(tree, i)
        loc <- if (nzchar(path)) path else "<root>"
        cli_abort(
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
#'   `.f(leaf_1, leaf_2, ..., leaf_n ...)`.
#' @seealso [map_tree()], [flatten()], [unflatten()]
#' @examples
#' pmap_tree(list(list(a = 1, b = 2), list(a = 10, b = 20)), `+`)
#' @export
pmap_tree <- function(.l, .f, ...) {
  if (!is.list(.l) || length(.l) == 0L) {
    cli_abort("{.arg .l} must be a non-empty list of trees.")
  }
  tree <- build_tree(.l[[1L]])
  for (i in seq_along(.l)[-1L]) {
    other <- build_tree(.l[[i]])
    if (!identical(tree, other)) {
      diff <- tree_diff(tree, other)
      header <- if (nzchar(diff$prefix)) {
        "First mismatch at {.code {diff$prefix}}:"
      } else {
        "Trees differ at the root:"
      }
      fmt_a <- format(diff$a)
      fmt_b <- format(diff$b)
      cli_abort(c(
        "All trees in {.arg .l} must have the same structure.",
        "x" = header,
        "*" = "{.code .l[[1]]}: {fmt_a}",
        "*" = "{.code .l[[{i}]]}: {fmt_b}"
      ))
    }
  }
  flats <- lapply(.l, flatten)
  n <- length(flats[[1L]])
  result <- lapply(seq_len(n), function(i) {
    tryCatch(
      do.call(.f, c(lapply(flats, `[[`, i), list(...))),
      error = function(e) {
        path <- tree_path(tree, i)
        loc <- if (nzchar(path)) path else "<root>"
        cli_abort(
          "Error applying {.arg .f} to leaves at {.code {loc}}.",
          parent = e,
          call = NULL
        )
      }
    )
  })
  unflatten(tree, result)
}

#' @title Difference Between Trees
#' @description
#' Walks two trees in parallel and returns the first path/subtree pair where
#' they diverge, or `NULL` if they are structurally identical. The returned
#' `prefix` follows [tree_path()] syntax.
#' @param a,b (`Node`)\cr
#'   Trees to compare, as returned by [build_tree()].
#' @param prefix (`character(1)`)\cr
#'   Path prefix accumulated during recursion; callers should leave as `""`.
#' @return `NULL` if `a` and `b` are structurally identical, otherwise a list
#'   with elements `prefix` (the path to the divergence), `a`, and `b` (the
#'   diverging subtrees).
#' @seealso [build_tree()], [tree_path()]
#' @keywords internal
tree_diff <- function(a, b, prefix = "") {
  if (identical(a, b)) {
    return(NULL)
  }
  tree_diff_impl(a, b, prefix)
}

# Recursive helper: assumes a and b are not identical and walks structurally
# without re-calling identical() at each level.
tree_diff_impl <- function(a, b, prefix) {
  # Handle class mismatch here so methods can assume `b` has the same class
  # as `a` -- UseMethod() only dispatches on `a`.
  if (!identical(class(a), class(b))) {
    return(list(prefix = prefix, a = a, b = b))
  }
  # Two empty nodes are structurally identical.
  if (inherits(a, "NullNode")) {
    return(NULL)
  }
  UseMethod("tree_diff_impl")
}

#' @export
tree_diff_impl.LeafNode <- function(a, b, prefix) {
  if (a$i == b$i) {
    return(NULL)
  }
  list(prefix = prefix, a = a, b = b)
}

#' @export
tree_diff_impl.ListNode <- function(a, b, prefix) {
  if (!identical(a$names, b$names) || length(a$nodes) != length(b$nodes)) {
    return(list(prefix = prefix, a = a, b = b))
  }
  for (j in seq_along(a$nodes)) {
    nm <- if (!is.null(a$names)) a$names[j] else ""
    suffix <- if (nzchar(nm)) {
      if (nzchar(prefix)) paste0("$", nm) else nm
    } else {
      paste0("[[", j, "]]")
    }
    d <- tree_diff_impl(a$nodes[[j]], b$nodes[[j]], paste0(prefix, suffix))
    if (!is.null(d)) {
      return(d)
    }
  }
  NULL
}

flat_mask_from_names <- function(tree, names) {
  if (is.null(names) || length(names) == 0L) {
    rep(TRUE, times = tree_size(tree))
  } else {
    mask <- tree$names %in% names
    rep(mask, times = vapply(tree$nodes, tree_size, integer(1L)))
  }
}
