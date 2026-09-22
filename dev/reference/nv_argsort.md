# Argsort

Returns the indices that would sort the array: over every element by
default, or along one axis. It is the index twin of
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md) and
takes the same `axis`, so the two always describe the same ordering.

## Usage

``` r
nv_argsort(x, axis = NULL, decreasing = FALSE, stable = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to compute the sort permutation. Negative values
  count from the end, i.e. `-1` refers to the last axis. If `NULL`
  (default), the input is first flattened to a 1-D array, like
  [`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md),
  and the indices refer to that flattening.

- decreasing:

  (`logical(1)`)  
  If `TRUE`, returns indices that produce a decreasing sort. Default
  `FALSE`.

- stable:

  (`logical(1)`)  
  If `TRUE`, the sort is stable: indices for equal values keep their
  original relative order. Default `FALSE`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) of
the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))  
Same shape as `x`, or 1-D holding every element's index when `axis` is
`NULL`. For a size-0 axis, the output is an empty array of the same
shape (a valid empty permutation). Indexing the flattened input by the
result reproduces
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md)'s
output.

## NaN handling

`NaN` values sort to the **end** (ascending) or **beginning**
(descending), regardless of sign. `+0` and `-0` compare equal.

## See also

[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md),
[`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md).

## Examples

``` r
x <- nv_array(c(3, 1, 4, 1, 5))
nv_argsort(x)
#> AnvlArray
#>  2
#>  4
#>  1
#>  3
#>  5
#> [ CPUi32{5} ] 

m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
nv_argsort(m) # indexes the flattened matrix
#> AnvlArray
#>  6
#>  2
#>  4
#>  1
#>  5
#>  3
#> [ CPUi32{6} ] 
nv_argsort(m, axis = 2L) # a permutation per row
#> AnvlArray
#>  2 1 3
#>  3 1 2
#> [ CPUi32{2,3} ] 
```
