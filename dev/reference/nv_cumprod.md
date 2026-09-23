# Cumulative Product

Cumulative product, optionally along a single axis. A boolean array is
multiplied as zeroes and ones, like
[`base::cumprod()`](https://rdrr.io/r/base/cumsum.html) does.

## Usage

``` r
nv_cumprod(x, axis = NULL, nan_rm = FALSE)
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
  [`base::cumprod()`](https://rdrr.io/r/base/cumsum.html).

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in float inputs. If `FALSE` (default),
  `NaN` propagates forward from its first occurrence. If `TRUE`, `NaN`
  is treated as the identity element of the cumulative op (`0` for sum,
  `1` for prod, `-Inf` / `+Inf` for max / min) and contributes nothing
  to the running value.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape when `axis` is given, and is 1-D of length
`prod(shape(x))` when `axis` is `NULL`, which flattens first. Has the
input's data type, except for a boolean input, which is accumulated at
the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).

## See also

[`prim_cumprod()`](https://r-xla.github.io/anvl/dev/reference/prim_cumprod.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
nv_cumprod(x)              # flatten, then accumulate
#> AnvlArray
#>    1
#>    2
#>    6
#>   24
#>  120
#>  720
#> [ CPUi32{6} ] 
nv_cumprod(x, axis = 1L)    # accumulate along rows
#> AnvlArray
#>   1  3  5
#>   2 12 30
#> [ CPUi32{2,3} ] 
nv_cumprod(nv_array(c(2, NaN, 3)))                # NaN propagates
#> AnvlArray
#>    2
#>  nan
#>  nan
#> [ CPUf32{3} ] 
nv_cumprod(nv_array(c(2, NaN, 3)), nan_rm = TRUE) # NaN treated as 1
#> AnvlArray
#>  2
#>  2
#>  6
#> [ CPUf32{3} ] 
```
