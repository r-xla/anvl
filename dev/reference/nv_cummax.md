# Cumulative Maximum

Running maximum, optionally along a single axis.

## Usage

``` r
nv_cummax(x, axis = NULL, indices = FALSE, nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to accumulate. Negative values count from the end,
  i.e. `-1` refers to the last axis. If `NULL` (default), the input is
  first flattened to a 1-D array, like
  [`base::cummax()`](https://rdrr.io/r/base/cumsum.html).

- indices:

  (`logical(1)`)  
  If `FALSE` (default), returns the running-maximum array. If `TRUE`,
  returns `list(values = ..., indices = ...)` where `indices` is the
  index of the last occurrence of the running maximum at each position,
  of the default integer data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  When `axis = NULL`, indices refer to the flattened input.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in float inputs. If `FALSE` (default),
  `NaN` propagates forward from its first occurrence. If `TRUE`, `NaN`
  is treated as the identity element of the cumulative op (`0` for sum,
  `1` for prod, `-Inf` / `+Inf` for max / min) and contributes nothing
  to the running value.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) \|
named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
One array when `indices = FALSE`, a named `list` of `values` and
`indices` when `indices = TRUE`. The values have the input's data type
and the indices the default integer data type; both have the input's
shape when `axis` is given, and are 1-D of length `prod(shape(x))` when
`axis` is `NULL`.

## See also

[`prim_cummax()`](https://r-xla.github.io/anvl/dev/reference/prim_cummax.md)
for the underlying primitive.

## Examples

``` r
# the running maximum keeps the data type; the indices are the default integer
x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
nv_cummax(x)
#> AnvlArray
#>  3
#>  3
#>  4
#>  4
#>  5
#>  9
#> [ CPUf32{6} ] 
nv_cummax(x, axis = 1L)
#> AnvlArray
#>  3 4 5
#>  3 4 9
#> [ CPUf32{2,3} ] 
nv_cummax(x, axis = 1L, indices = TRUE)
#> $values
#> AnvlArray
#>  3 4 5
#>  3 4 9
#> [ CPUf32{2,3} ] 
#> 
#> $indices
#> AnvlArray
#>  1 1 1
#>  1 1 2
#> [ CPUi32{2,3} ] 
#> 
nv_cummax(nv_array(c(1, NaN, 3)))                # NaN propagates
#> AnvlArray
#>    1
#>  nan
#>  nan
#> [ CPUf32{3} ] 
nv_cummax(nv_array(c(1, NaN, 3)), nan_rm = TRUE) # NaN skipped
#> AnvlArray
#>  1
#>  1
#>  3
#> [ CPUf32{3} ] 
```
