# Cumulative Product

Cumulative product, optionally along a single dimension.

## Usage

``` r
nv_cumprod(x, dim = NULL, nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to accumulate. If `NULL` (default), the input is
  first flattened to a 1-D array, like
  [`base::cumprod()`](https://rdrr.io/r/base/cumsum.html).

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates forward from its first occurrence. If
  `TRUE`, `NaN` is treated as the identity element of the cumulative op
  (`0` for sum, `1` for prod, `-Inf` / `+Inf` for max / min) and
  contributes nothing to the running value.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## Relation to base R

Both `nv_cumprod()` (with `dim = NULL`) and
[`base::cumprod()`](https://rdrr.io/r/base/cumsum.html) flatten a
multi-dimensional input to 1-D before accumulating, but the flatten
order differs: anvl arrays are row-major (C order), so the flattened
sequence iterates the last dim fastest, whereas base R uses column-major
(Fortran) order. The two agree on 1-D inputs.

## See also

[`prim_cumprod()`](https://r-xla.github.io/anvl/dev/reference/prim_cumprod.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
nv_cumprod(x)              # row-major flatten, then accumulate
#> AnvlArray
#>    1
#>    3
#>   15
#>   30
#>  120
#>  720
#> [ CPUi32{6} ] 
nv_cumprod(x, dim = 1L)    # accumulate along rows
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
