# map_tree / reports the leaf path on error

    Code
      map_tree(list(a = 1, b = list(c = 2)), function(x) if (x == 2) cli::cli_abort(
        "boom") else x)
    Condition
      Error:
      ! Error applying `.f` to leaf at `b$c`.
      Caused by error in `.f()`:
      ! boom

# pmap_tree / errors when trees have different structure

    Code
      pmap_tree(list(list(model = list(weights = list(a = 1), bias = 2)), list(model = list(
        weights = list(a = 1), bias = list(z = 9)))), `+`)
    Condition
      Error in `pmap_tree()`:
      ! All trees in `.l` must have the same structure.
      x First mismatch at `model$bias`:
      * `.l[[1]]`: *
      * `.l[[2]]`: list(z = *)

# tree node formatting / renders LeafNode and ListNode as R-idiomatic literals

    Code
      build_tree(1)
    Output
      *
    Code
      build_tree(list())
    Output
      list()
    Code
      build_tree(list(1, 2))
    Output
      list(*, *)
    Code
      build_tree(list(a = 1, b = 2))
    Output
      list(a = *, b = *)
    Code
      build_tree(list(1, b = 2))
    Output
      list(*, b = *)
    Code
      build_tree(list(a = list(b = 1, c = 2), d = 3))
    Output
      list(a = list(b = *, c = *), d = *)

# tree_diff / locates the first divergence

    Code
      tree_diff(build_tree(1), build_tree(list(1, 2)))
    Output
      $prefix
      [1] ""
      
      $a
      *
      
      $b
      list(*, *)
      
    Code
      tree_diff(build_tree(list(a = 1, b = 2)), build_tree(list(p = 1, q = 2)))
    Output
      $prefix
      [1] ""
      
      $a
      list(a = *, b = *)
      
      $b
      list(p = *, q = *)
      
    Code
      tree_diff(build_tree(list(1, 2)), build_tree(list(1, 2, 3)))
    Output
      $prefix
      [1] ""
      
      $a
      list(*, *)
      
      $b
      list(*, *, *)
      
    Code
      tree_diff(build_tree(list(list(a = 1), list(a = 1))), build_tree(list(list(a = 1),
      list(a = 1, b = 2))))
    Output
      $prefix
      [1] "[[2]]"
      
      $a
      list(a = *)
      
      $b
      list(a = *, b = *)
      
    Code
      tree_diff(build_tree(list(pair = list(list(a = 1), 0))), build_tree(list(pair = list(
        list(a = 1), list(c = 0)))))
    Output
      $prefix
      [1] "pair[[2]]"
      
      $a
      *
      
      $b
      list(c = *)
      
    Code
      tree_diff(build_tree(list(a = 1, b = 2)), build_tree(list(a = 1, b = 2)))
    Output
      NULL

# tree_path

    Code
      tree_path(build_tree(list(x = 1)), 1L)
    Output
      [1] "x"
    Code
      tree_path(build_tree(list(l = list(a = 1, b = 2))), 1L)
    Output
      [1] "l$a"
    Code
      tree_path(build_tree(list(l = list(a = 1, b = 2))), 2L)
    Output
      [1] "l$b"
    Code
      tree_path(build_tree(list(l = list(1, 2))), 2L)
    Output
      [1] "l[[2]]"
    Code
      tree_path(build_tree(list(l = list(1, b = 2))), 1L)
    Output
      [1] "l[[1]]"
    Code
      tree_path(build_tree(list(l = list(1, b = 2))), 2L)
    Output
      [1] "l$b"
    Code
      tree_path(build_tree(list(l = list(list(a = 1)))), 1L)
    Output
      [1] "l[[1]]$a"
    Code
      tree_path(build_tree(list(x = 1, y = 2)), 2L)
    Output
      [1] "y"
    Code
      tree_path(build_tree(list(pair = list(list(a = 1), list(b = 2)))), 2L)
    Output
      [1] "pair[[2]]$b"

