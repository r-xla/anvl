# Variance Reduction

Computes the variance along the specified axes.

## Usage

``` r
nv_var(x, axes = NULL, drop = TRUE, correction = 1L, nan_rm = FALSE)
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

- correction:

  (`integer(1)`)  
  Degrees of freedom correction. Default is `1` (Bessel's correction).

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
axes are removed. When `drop = FALSE`, the reduced axes are set to 1.

## Details

Uses Bessel's correction by default (`correction = 1`), matching R's
[`var()`](https://rdrr.io/r/stats/cor.html). Set `correction = 0` for
population variance.

## See also

[`nv_sd()`](https://r-xla.github.io/anvl/dev/reference/nv_sd.md),
[`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3, 4, 5))
nv_var(x)             # all axes -> scalar
#> AnvlArray
#>  2.5000
#> [ CPUf32{} ] 
nv_var(x, axes = 1L)
#> AnvlArray
#>  2.5000
#> [ CPUf32{} ] 
nv_var(nv_array(c(1, NaN, 3, 5)), axes = 1L, nan_rm = TRUE)
#> AnvlArray
#>  4
#> [ CPUf32{} ] 
```
