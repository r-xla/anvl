# Mean

Computes the arithmetic mean along the specified dimensions. You can
also use [`mean()`](https://rdrr.io/r/base/mean.html).

## Usage

``` r
nv_mean(x, dims = NULL, drop = TRUE, nan_rm = FALSE)

# S3 method for class 'AnvlArray'
mean(x, trim = 0, na.rm = FALSE, ..., dims = NULL, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

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

- trim:

  Currently not supported.

- na.rm:

  Forwarded to `nv_mean()`'s `nan_rm` argument.

- ...:

  No additional arguments.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
dimensions are removed. When `drop = FALSE`, the reduced dimensions are
set to 1.

## See also

[`nv_reduce_sum()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_sum.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
nv_mean(x)            # all dims -> scalar
#> AnvlArray
#>  3.5000
#> [ CPUf32?{} ] 
nv_mean(x, dims = 1L)
#> AnvlArray
#>  1.5000
#>  3.5000
#>  5.5000
#> [ CPUf32?{3} ] 
nv_mean(nv_array(c(1, NaN, 3)))
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_mean(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#> AnvlArray
#>  2
#> [ CPUf32{} ] 
```
