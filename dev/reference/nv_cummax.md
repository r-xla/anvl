# Cumulative Maximum

Running maximum, optionally along a single axis.

## Usage

``` r
nv_cummax(x, axis = NULL, indices = FALSE, nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

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
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates forward from its first occurrence. If
  `TRUE`, `NaN` is treated as the identity element of the cumulative op
  (`0` for sum, `1` for prod, `-Inf` / `+Inf` for max / min) and
  contributes nothing to the running value.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
(when `indices = FALSE`) or named list of two arrays (when
`indices = TRUE`).

## Relation to base R

`nv_cummax()` with `axis = NULL` and
[`base::cummax()`](https://rdrr.io/r/base/cumsum.html) both flatten
first, but in different orders – anvl arrays are row-major (C order),
base R is column-major (Fortran) – so for a multi-axis input the two
give different running values. They agree on 1-D inputs.

## See also

[`prim_cummax()`](https://r-xla.github.io/anvl/dev/reference/prim_cummax.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
nv_cummax(x)
#> AnvlArray
#>  3
#>  4
#>  5
#>  5
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
