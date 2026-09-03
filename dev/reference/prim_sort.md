# Primitive Sort

Sorts arrays along the given axis.

Sorting is determined by the *first array* only: it is the sort key, and
any additional arrays are reordered with the same permutation that sorts
the first. This enables idioms like *argsort* (sort `x` paired with an
`iota` and read off the second output) and key-value sorts (sort `keys`
paired with `values`).

All arrays must have the same shape; their dtypes may differ.
1-dimensional slices along `axis` are sorted independently; other axes
are preserved.

## Usage

``` r
prim_sort(xs, axis = 1L, descending = FALSE, is_stable = FALSE)
```

## Arguments

- xs:

  (`list` of
  [`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One or more arrays to sort. The first is the sort key; the rest are
  carried along under the same permutation. All must share the same
  shape.

- axis:

  (`integer(1)`)  
  Axis along which to sort. Negative values count from the end, i.e.
  `-1` refers to the last axis.

- descending:

  (`logical(1)`)  
  If `TRUE`, sort the key in descending order (largest first). Default
  `FALSE`. Additional arrays are reordered by the same permutation
  regardless.

- is_stable:

  (`logical(1)`)  
  If `TRUE`, the sort is stable: the relative order of equal *keys* is
  preserved. Default `FALSE`.

## Value

`list` of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
One sorted output per element of `xs`, in the same order. Each output
has the same shape and data type as the corresponding input.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_sort()`](https://r-xla.github.io/stablehlo/reference/hlo_sort.html)
with a comparator that uses
[`hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html)
(`LT` for ascending, `GT` for descending) on the first array. For float
keys the comparator uses `compare_type = "TOTALORDER"` and canonicalizes
`-0`/`+0` and `-NaN`/`+NaN` to their positive form before comparing, so
all `NaN` values land at one end of the result regardless of sign.
Integer keys use `SIGNED` / `UNSIGNED` as appropriate.

## See also

[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md),
[`nv_argsort()`](https://r-xla.github.io/anvl/dev/reference/nv_argsort.md),
[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
[`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md)

## Examples

``` r
x <- nv_array(c(3, 1, 4, 1, 5))
prim_sort(list(x), axis = 1L)[[1L]]
#> AnvlArray
#>  1
#>  1
#>  3
#>  4
#>  5
#> [ CPUf32{5} ] 

# Sort indices by the values (argsort): pair x with iota and read off
# the second result.
idx <- nv_iota(axis = 1L, dtype = "i64", shape = 5L)
out <- prim_sort(list(x, idx), axis = 1L)
out[[1L]] # sorted x
#> AnvlArray
#>  1
#>  1
#>  3
#>  4
#>  5
#> [ CPUf32{5} ] 
out[[2L]] # permutation indices
#> AnvlArray
#>  2
#>  4
#>  1
#>  3
#>  5
#> [ CPUi64{5} ] 
```
