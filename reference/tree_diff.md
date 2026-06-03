# Difference Between Trees

Walks two trees in parallel and returns the first path/subtree pair
where they diverge, or `NULL` if they are structurally identical. The
returned `prefix` follows
[`tree_path()`](https://r-xla.github.io/anvl/reference/tree_path.md)
syntax.

## Usage

``` r
tree_diff(a, b, prefix = "")
```

## Arguments

- a, b:

  (`Node`)  
  Trees to compare, as returned by
  [`build_tree()`](https://r-xla.github.io/anvl/reference/build_tree.md).

- prefix:

  (`character(1)`)  
  Path prefix accumulated during recursion; callers should leave as
  `""`.

## Value

`NULL` if `a` and `b` are structurally identical, otherwise a list with
elements `prefix` (the path to the divergence), `a`, and `b` (the
diverging subtrees).

## See also

[`build_tree()`](https://r-xla.github.io/anvl/reference/build_tree.md),
[`tree_path()`](https://r-xla.github.io/anvl/reference/tree_path.md)
