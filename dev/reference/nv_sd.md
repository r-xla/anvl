# Standard Deviation

Computes the standard deviation along the specified axes.

## Usage

``` r
nv_sd(x, axes = NULL, drop = TRUE, correction = 1L, nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reduce over. Negative values count from the end, i.e. `-1`
  refers to the last axis. If `NULL` (default), reduces over all axes.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced axes: removed from the output shape if
  `TRUE`, set to 1 if `FALSE`.

- correction:

  (`integer(1)`)  
  Degrees of freedom correction. Default is `1` (Bessel's correction).

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in float inputs. If `FALSE` (default),
  `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type where that is a float, and the default float
data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
otherwise. The shape is the input's with the reduced axes removed
(`drop = TRUE`) or set to 1 (`drop = FALSE`).

## Details

Uses Bessel's correction by default (`correction = 1`), matching R's
[`sd()`](https://rdrr.io/r/stats/sd.html). Set `correction = 0` for
population standard deviation.

## See also

[`nv_var()`](https://r-xla.github.io/anvl/dev/reference/nv_var.md),
[`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md)

## Examples

``` r
x <- nv_array(1:5)
# the result is a float, even though the input is an integer
nv_sd(x)
#> AnvlArray
#>  1.5811
#> [ CPUf32{} ] 

# Bessel's correction by default, correction = 0 for the population value
nv_sd(x, correction = 0L)
#> AnvlArray
#>  1.4142
#> [ CPUf32{} ] 
```
