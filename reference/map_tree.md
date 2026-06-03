# Map Over a Tree

Apply a function to each leaf of a (possibly nested) list, preserving
the tree structure. Equivalent to flattening `.x` with
[`flatten()`](https://r-xla.github.io/anvl/reference/flatten.md),
applying `.f` to each leaf, and reassembling with
[`unflatten()`](https://r-xla.github.io/anvl/reference/unflatten.md).

## Usage

``` r
map_tree(.x, .f, ...)
```

## Arguments

- .x:

  (any)  
  A leaf or a (nested) list of leaves.

- .f:

  (`function`)  
  Function to apply to each leaf of `.x`.

- ...:

  Additional arguments passed to `.f`.

## Value

An object with the same nesting structure as `.x`, where each leaf is
the result of `.f(leaf, ...)`.

## See also

[`flatten()`](https://r-xla.github.io/anvl/reference/flatten.md),
[`build_tree()`](https://r-xla.github.io/anvl/reference/build_tree.md),
[`unflatten()`](https://r-xla.github.io/anvl/reference/unflatten.md),
[`await()`](https://r-xla.github.io/anvl/reference/await.md)

## Examples

``` r
map_tree(list(a = 1, b = list(c = 2, d = 3)), \(x) x + 1)
#> $a
#> [1] 2
#> 
#> $b
#> $b$c
#> [1] 3
#> 
#> $b$d
#> [1] 4
#> 
#> 
```
