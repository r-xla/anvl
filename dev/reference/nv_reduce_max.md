# Max Reduction

Finds the maximum of array elements along the specified axes.

## Usage

``` r
nv_reduce_max(x, axes = NULL, drop = TRUE, nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reduce. Negative values count from the end, i.e. `-1` refers
  to the last axis. If `NULL` (default), reduces over all axes,
  returning a scalar.

- drop:

  (`logical(1)`)  
  Whether to drop reduced axes.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
axes are removed. When `drop = FALSE`, the reduced axes are set to 1.

## The [`max()`](https://rdrr.io/r/base/Extremes.html) generic

[`max()`](https://rdrr.io/r/base/Extremes.html) reduces over all axes
and, like [`base::max()`](https://rdrr.io/r/base/Extremes.html), takes
several data arguments: `max(x, y)` is the largest element of both
arrays. `na.rm` becomes `nan_rm`. Beyond base R, named arguments are
passed on, so `max(x, axes = 1L)` reduces a single axis – but only when
`x` is the only data argument.

## See also

[`prim_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_max.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
nv_reduce_max(x)            # all axes -> scalar
#> AnvlArray
#>  6
#> [ CPUi32{} ] 
nv_reduce_max(x, axes = 1L)
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUi32{3} ] 
nv_reduce_max(nv_array(c(1, NaN, 3)))
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_reduce_max(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
```
