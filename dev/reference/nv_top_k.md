# Top-K Elements

Returns the `k` largest values over one or more axes, sorted in
decreasing order.

## Usage

``` r
nv_top_k(x, k, axes = NULL, indices = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with at least 1 axis. Can be any numeric data type. An R
  value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- k:

  (`integer(1)`)  
  Number of top elements to return. Must be a whole number between 1 and
  the number of elements `axes` holds together; a fractional or logical
  `k` is refused rather than truncated.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to take the top `k` over. Negative values count from the end,
  i.e. `-1` refers to the last axis. If `NULL` (default), ranks over
  every axis.

- indices:

  (`logical(1)`)  
  If `FALSE` (default), returns just the top-`k` values. If `TRUE`,
  returns `list(values = ..., indices = ...)` where `indices` holds the
  position of each top-`k` value.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) \|
named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
One array when `indices = FALSE`, a named `list` of `values` and
`indices` when `indices = TRUE`. The values have the input's data type
and the indices the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
Both have the input's shape with `axes` replaced by a single axis of
size `k`, sitting where the first of them was; values are sorted
decreasing along that axis.

## Ranking several axes

Taking the top `k` over several axes ranks all of their elements
together, so `nv_top_k(x, k)` equals `nv_top_k(nv_flatten(x), k)`. The
indices then index the column-major flattening of those axes – the order
[`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
produces – rather than any single axis.

## NaN handling

`NaN` ranks larger than any finite value (so it appears first in the
top-`k` output); `-NaN` ranks smaller. Unlike
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md),
the sign bit is not canonicalized.

## See also

[`prim_top_k()`](https://r-xla.github.io/anvl/dev/reference/prim_top_k.md)
for the underlying primitive,
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md).

## Examples

``` r
# the values keep the input's data type, the indices the default integer
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
nv_top_k(x, k = 3L)
#> AnvlArray
#>  9
#>  6
#>  5
#> [ CPUf32{3} ] 
nv_top_k(x, k = 3L, indices = TRUE)
#> $values
#> AnvlArray
#>  9
#>  6
#>  5
#> [ CPUf32{3} ] 
#> 
#> $indices
#> AnvlArray
#>  6
#>  8
#>  5
#> [ CPUi32{3} ] 
#> 

m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
nv_top_k(m, k = 2L, axes = 2L) # the top 2 of each row
#> AnvlArray
#>  5 3
#>  4 2
#> [ CPUf32{2,2} ] 
nv_top_k(m, k = 2L) # the top 2 of the whole matrix
#> AnvlArray
#>  5
#>  4
#> [ CPUf32{2} ] 
```
