# Normal Density

Computes the probability density function of the normal distribution:
\$\$f(x) = \frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\$\$ where \\\mu\\ is the
mean and \\\sigma\\ is the standard deviation of the distribution.
Converts `mean` and `sd` to the data type of `x`.

## Usage

``` r
nv_dnorm(x, mean = 0, sd = 1, log = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Quantiles at which to evaluate the density.

- mean:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Mean of the distribution (scalar or same shape as `x`).

- sd:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Standard deviation of the distribution (scalar or same shape as `x`).
  Must be positive, otherwise results are invalid.

- log:

  (`logical(1)`)  
  If `TRUE`, returns the log-density instead of the density.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_rnorm.md)
for sampling from a normal distribution.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
nv_dnorm(x)
#> AnvlArray
#>  0.2420
#>  0.3989
#>  0.2420
#> [ CPUf32{3} ] 
nv_dnorm(x, mean = 1, sd = 2)
#> AnvlArray
#>  0.1210
#>  0.1760
#>  0.1995
#> [ CPUf32{3} ] 
nv_dnorm(x, log = TRUE)
#> AnvlArray
#>  -1.4189
#>  -0.9189
#>  -1.4189
#> [ CPUf32{3} ] 
```
