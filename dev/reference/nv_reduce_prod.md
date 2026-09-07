# Product Reduction

Multiplies array elements along the specified axes. A boolean array is
multiplied as zeroes and ones, like
[`base::prod()`](https://rdrr.io/r/base/prod.html) does.

## Usage

``` r
nv_reduce_prod(x, axes = NULL, drop = TRUE, nan_rm = FALSE)
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
Has the same data type as the input, except for a boolean input, which
is accumulated at `i32`: `TRUE` counts as one, rather than being folded
with a logical or/and. When `drop = TRUE`, the reduced axes are removed.
When `drop = FALSE`, the reduced axes are set to 1.

## See also

[`prim_reduce_prod()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_prod.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
nv_reduce_prod(x)            # all axes -> scalar
#> AnvlArray
#>  720
#> [ CPUi32{} ] 
nv_reduce_prod(x, axes = 1L)
#> AnvlArray
#>   2
#>  12
#>  30
#> [ CPUi32{3} ] 
nv_reduce_prod(nv_array(c(2, NaN, 3)))
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_reduce_prod(nv_array(c(2, NaN, 3)), nan_rm = TRUE)
#> AnvlArray
#>  6
#> [ CPUf32{} ] 
```
