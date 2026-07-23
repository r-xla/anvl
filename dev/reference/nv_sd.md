# Standard Deviation Reduction

Computes the standard deviation along the specified axes.

## Usage

``` r
nv_sd(x, axes = NULL, drop = TRUE, correction = 1L, nan_rm = FALSE)
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
[`sd()`](https://rdrr.io/r/stats/sd.html). Set `correction = 0` for
population standard deviation.

## See also

[`nv_var()`](https://r-xla.github.io/anvl/dev/reference/nv_var.md),
[`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3, 4, 5))
nv_sd(x)              # all axes -> scalar
#> AnvlArray
#>  1.5811
#> [ CPUf32{} ] 
nv_sd(x, axes = 1L)
#> AnvlArray
#>  1.5811
#> [ CPUf32{} ] 
```
