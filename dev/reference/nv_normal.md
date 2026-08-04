# The Normal Distribution

Density (`nv_dnorm`), distribution function (`nv_pnorm`) and random
generation (`nv_rnorm`) for the Normal distribution with mean `mean` and
standard deviation `sd`.

## Usage

``` r
nv_dnorm(x, mean = 0, sd = 1, log = FALSE)

nv_pnorm(q, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE)

nv_rnorm(shape, initial_state, dtype = "f32", mean = 0, sd = 1)
```

## Arguments

- x, q:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Quantiles at which to evaluate the density (`x`) or the distribution
  function (`q`).

- mean:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Mean of the distribution. For `nv_dnorm`/`nv_pnorm` a scalar or the
  same shape as `x`/`q`; for `nv_rnorm` a scalar or the same shape as
  `shape`.

- sd:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Standard deviation of the distribution. For `nv_dnorm`/`nv_pnorm` a
  scalar or the same shape as `x`/`q`; for `nv_rnorm` a scalar or the
  same shape as `shape`. Must be positive. Because `sd` is arrayish, it
  may be a traced value whose contents are unknown until the compiled
  program runs, so this is never checked: a non-positive `sd` silently
  produces garbage (`NaN` or nonsense values) rather than raising an
  error.

- log, log_p:

  (`logical(1)`)  
  If `TRUE`, the densities/probabilities are given as logarithms.

- lower_tail:

  (`logical(1)`)  
  If `TRUE` (default), probabilities are \\P(X \le q)\\; otherwise,
  \\P(X \> q)\\.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  RNG state (`ui64[2]`).

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

## Value

`nv_dnorm()` and `nv_pnorm()` return an
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
with the same shape and data type as `x`/`q`.

`nv_rnorm()` returns a [`list()`](https://rdrr.io/r/base/list.html) of
two [`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
elements: the updated RNG state and the sampled values.

## Details

The Normal distribution has probability density function: \$\$f(x) =
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\$\$ where \\\mu\\ is the
mean and \\\sigma\\ is the standard deviation. The `mean` and `sd` are
converted to the data type of `x`/`q`.

`nv_pnorm` uses the asymptotic expansion from Abramowitz & Stegun
(1964), equation 26.2.12, in the left tail when `log_p = TRUE` to
maintain accuracy.

## Random generation

`nv_rnorm` samples via the Box-Muller transform. To sample with a
covariance structure, use a Cholesky decomposition.

`mean` and `sd` are
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md), so
they may vary across the sample: they are applied to the draws after
they have been reshaped to `shape`, and so may either be scalars or have
exactly that shape.

## References

Abramowitz M, Stegun I (1964). *Handbook of Mathematical Functions with
Formulas, Graphs, and Mathematical Tables*, number 55 series Applied
Mathematics Series. Dover Publications, New York. ISBN 0-486-61272-4.

## See also

Other rng:
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md),
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)

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
state <- nv_rng_state(42L)
result <- nv_rnorm(c(2, 3), state)
result[[2]]
#> AnvlArray
#>  -0.0675  0.9489  1.9457
#>  -0.5255  1.2002  0.0008
#> [ CPUf32{2,3} ] 

# `sd` may also be an array of the same shape as the sample
sds <- nv_array(matrix(c(0.01, 0.1, 1, 10, 100, 1000), nrow = 2))
nv_rnorm(c(2, 3), state, sd = sds)[[2]]
#> AnvlArray
#>   -0.0007   0.9489 194.5720
#>   -0.0526  12.0017   0.7665
#> [ CPUf32{2,3} ] 
```
