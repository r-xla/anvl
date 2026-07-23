# Mean

Computes the arithmetic mean along the specified axes. You can also use
[`mean()`](https://rdrr.io/r/base/mean.html).

## Usage

``` r
nv_mean(x, axes = NULL, drop = TRUE, nan_rm = FALSE)

# S3 method for class 'AnvlArray'
mean(x, trim = 0, na.rm = FALSE, ..., axes = NULL, drop = TRUE)
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

- trim:

  Currently not supported.

- na.rm:

  Forwarded to `nv_mean()`'s `nan_rm` argument.

- ...:

  No additional arguments.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
axes are removed. When `drop = FALSE`, the reduced axes are set to 1.

## See also

[`nv_reduce_sum()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_sum.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
nv_mean(x)            # all axes -> scalar
#> AnvlArray
#>  3.5000
#> [ CPUf32?{} ] 
nv_mean(x, axes = 1L)
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
