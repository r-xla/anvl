# The Normal Distribution

Density (`nv_dnorm`) and distribution function (`nv_pnorm`) of the
Normal distribution with mean `mean` and standard deviation `sd`.

## Usage

``` r
nv_dnorm(x, mean = 0, sd = 1, log = FALSE)

nv_pnorm(q, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE)
```

## Arguments

- x, q:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Quantiles at which to evaluate the density (`x`) or the distribution
  function (`q`).

- mean:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Mean of the distribution (scalar or same shape as `x`/`q`).

- sd:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Standard deviation of the distribution (scalar or same shape as
  `x`/`q`). Must be positive, otherwise results are invalid.

- log, log_p:

  (`logical(1)`)  
  If `TRUE`, the densities/probabilities are given as logarithms.

- lower_tail:

  (`logical(1)`)  
  If `TRUE` (default), probabilities are \\P(X \le q)\\; otherwise,
  \\P(X \> q)\\.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## Details

The Normal distribution has probability density function: \$\$f(x) =
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\$\$ where \\\mu\\ is the
mean and \\\sigma\\ is the standard deviation. The `mean` and `sd` are
converted to the data type of `x`/`q`.

`nv_pnorm` uses the asymptotic expansion from Abramowitz & Stegun
(1964), equation 26.2.12, in the left tail when `log_p = TRUE` to
maintain accuracy.

## References

Abramowitz M, Stegun I (1964). *Handbook of Mathematical Functions with
Formulas, Graphs, and Mathematical Tables*, number 55 series Applied
Mathematics Series. Dover Publications, New York. ISBN 0-486-61272-4.

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

nv_pnorm(x)
#> AnvlArray
#>  0.1587
#>  0.5000
#>  0.8413
#> [ CPUf32{3} ] 
nv_pnorm(x, mean = 1, sd = 2)
#> AnvlArray
#>  0.1587
#>  0.3085
#>  0.5000
#> [ CPUf32{3} ] 
nv_pnorm(x, lower_tail = FALSE)
#> AnvlArray
#>  0.8413
#>  0.5000
#>  0.1587
#> [ CPUf32{3} ] 
nv_pnorm(x, log_p = TRUE)
#> AnvlArray
#>  -1.8410
#>  -0.6931
#>  -0.1728
#> [ CPUf32{3} ] 
```
