# Argsort

Returns the indices that would sort the array along an axis.

## Usage

``` r
nv_argsort(x, axis = NULL, decreasing = FALSE, stable = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to compute the sort permutation. Negative values
  count from the end, i.e. `-1` refers to the last axis. If `NULL`
  (default), uses the last axis.

- decreasing:

  (`logical(1)`)  
  If `TRUE`, returns indices that produce a decreasing sort. Default
  `FALSE`.

- stable:

  (`logical(1)`)  
  If `TRUE`, the sort is stable: indices for equal values keep their
  original relative order. Default `FALSE`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) of
dtype `i32`  
Same shape as `x`. For a size-0 axis, the output is an empty `i32` array
of the same shape (a valid empty permutation).
`as_array(x)[as_array(nv_argsort(x))]` reproduces the sorted array (for
1-D inputs).

## NaN handling

`NaN` values sort to the **end** (ascending) or **beginning**
(descending), regardless of sign. `+0` and `-0` compare equal.

## See also

[`nv_sort()`](https://r-xla.github.io/anvl/reference/nv_sort.md),
[`prim_sort()`](https://r-xla.github.io/anvl/reference/prim_sort.md).

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
```
