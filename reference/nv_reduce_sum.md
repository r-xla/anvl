# Sum Reduction

Sums array elements along the specified dimensions.

## Usage

``` r
nv_reduce_sum(operand, dims = NULL, drop = TRUE, nan_rm = FALSE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Dimensions to reduce. If `NULL` (default), reduces over all
  dimensions, returning a scalar.

- drop:

  (`logical(1)`)  
  Whether to drop reduced dimensions.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
dimensions are removed. When `drop = FALSE`, the reduced dimensions are
set to 1.

## See also

[`prim_reduce_sum()`](https://r-xla.github.io/anvl/reference/prim_reduce_sum.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
nv_reduce_sum(x)            # all dims -> scalar
#> AnvlArray
#>  21
#> [ CPUi32{} ] 
nv_reduce_sum(x, dims = 1L)
#> AnvlArray
#>   3
#>   7
#>  11
#> [ CPUi32{3} ] 
nv_reduce_sum(nv_array(c(1, NaN, 3)))
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_reduce_sum(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#> AnvlArray
#>  4
#> [ CPUf32{} ] 
```
