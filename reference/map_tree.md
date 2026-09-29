# Map a Function over Trees

Apply a function to each leaf of a (possibly nested) list, keeping its
structure, e.g. to a list of arrays a jitted function returns.

- `map_tree()` maps over one tree.

- `pmap_tree()` maps over several trees of the same structure in
  parallel, calling `.f` with one leaf from each.

## Usage

``` r
map_tree(.x, .f, ...)

pmap_tree(.l, .f, ...)
```

## Arguments

- .x:

  (any)  
  A leaf or a (nested) list of leaves.

- .f:

  (`function`)  
  Function to apply. `map_tree()` calls it with each leaf of `.x`,
  `pmap_tree()` with one leaf from each tree in `.l`, in order.

- ...:

  Additional arguments passed to `.f` after the leaves.

- .l:

  (`list`)  
  A non-empty list of trees, all with the same structure.

## Value

A tree with the same structure as `.x` (or `.l[[1]]`), where each leaf
is the result of `.f`.

## Details

These are implemented in
[`pjrt::map_tree()`](https://r-xla.github.io/pjrt/reference/map_tree.html)
and
[`pjrt::pmap_tree()`](https://r-xla.github.io/pjrt/reference/pmap_tree.html).

## Examples

``` r
out <- list(a = nv_array(1:2), b = list(c = nv_scalar(3)))
map_tree(out, dtype)
#> $a
#> <i32>
#> 
#> $b
#> $b$c
#> <f32>
#> 
#> 
map_tree(out, as_array)
#> $a
#> [1] 1 2
#> 
#> $b
#> $b$c
#> [1] 3
#> 
#> 
pmap_tree(list(list(a = 1, b = 2), list(a = 10, b = 20)), `+`)
#> $a
#> [1] 11
#> 
#> $b
#> [1] 22
#> 
```
