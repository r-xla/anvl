# Map Over Multiple Trees

Apply a function leaf-wise over several trees with the same structure.
All trees in `.l` must have identical structure.

## Usage

``` r
pmap_tree(.l, .f, ...)
```

## Arguments

- .l:

  (`list`)  
  A non-empty list of trees, all with the same structure.

- .f:

  (`function`)  
  Function to call with one leaf from each tree (positional arguments,
  in the order given by `.l`).

- ...:

  Additional arguments passed to `.f` after the per-tree leaves.

## Value

A tree with the same structure as `.l[[1]]`, where each leaf is
`.f(leaf_1, leaf_2, ..., leaf_n ...)`.

## See also

[`map_tree()`](https://r-xla.github.io/anvl/reference/map_tree.md),
[`flatten()`](https://r-xla.github.io/anvl/reference/flatten.md),
[`unflatten()`](https://r-xla.github.io/anvl/reference/unflatten.md)

## Examples

``` r
pmap_tree(list(list(a = 1, b = 2), list(a = 10, b = 20)), `+`)
#> $a
#> [1] 11
#> 
#> $b
#> [1] 22
#> 
```
