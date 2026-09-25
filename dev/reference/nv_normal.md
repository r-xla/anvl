# The Normal Distribution

Density (`nv_dnorm`), distribution function (`nv_pnorm`), quantile
function (`nv_qnorm`), and random generation (`nv_rnorm`) for the Normal
distribution with mean `mean` and standard deviation `sd`.

## Usage

``` r
nv_dnorm(x, mean = 0, sd = 1, log = FALSE)

nv_pnorm(q, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE)

nv_qnorm(p, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE)

nv_rnorm(shape, state, mean = 0, sd = 1, dtype = NULL)
```

## Arguments

- x, q:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Quantiles at which to evaluate the density (`x`) or the distribution
  function (`q`). `x` can be any float data type; `q` must be `f32` or
  `f64` (see "Details"). An R `double` is materialized at its [default
  data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- mean:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Mean of the distribution, scalar or the same shape as `x`/`q`/`p`.
  Converted to the argument's data type.

- sd:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Standard deviation of the distribution, scalar or the same shape as
  `x`/`q`/`p`. Converted to the argument's data type. Must be positive,
  otherwise results are invalid.

- log, log_p:

  (`logical(1)`)  
  If `TRUE`, the densities/probabilities are given as logarithms. For
  `nv_qnorm` this describes the input `p`.

- lower_tail:

  (`logical(1)`)  
  If `TRUE` (default), probabilities are \\P(X \le x)\\; otherwise,
  \\P(X \> x)\\.

- p:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Probabilities at which to evaluate the quantile function. Values
  outside \\\[0, 1\]\\ give `NaN`. Must be `f32` or `f64` (see
  "Details"); `mean` and `sd` are converted to it. An R value
  materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the result.

- state:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  RNG state: a 1-D array of two `ui64` elements, as
  [`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md)
  returns. The data type and length are fixed by the generator, not by
  the default data types, and the returned `state` has them too.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Floating point data type. The default (`NULL`) uses the [default float
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) \|
named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
`nv_dnorm()`, `nv_pnorm()` and `nv_qnorm()` return an
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
with the shape and data type of `x`/`q`/`p`.

`nv_rnorm()` returns a named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md):
`state`, the updated RNG state with the input `state`'s data type and
shape, and `values`, the sample of shape `shape` and the data type
described under `dtype`.

## Details

The Normal distribution has probability density function: \$\$f(x) =
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\$\$ where \\\mu\\ is the
mean and \\\sigma\\ is the standard deviation.

`nv_pnorm` uses the asymptotic expansion from Abramowitz & Stegun
(1964), equation 26.2.12, in the left tail when `log_p = TRUE` to
maintain accuracy.

`nv_qnorm` uses the same minimax rational approximation as Moshier
(1989) (this is `ndtri` in the Cephes library as used by JAX) for `f64`,
and uses a new lower degree Remez minimax rational approximation on the
same intervals for `f32`.

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

Moshier S (1989). *Methods and Programs for Mathematical Functions*.
Ellis Horwood. ISBN 0-7458-0289-3.

## See also

`nv_rnorm()` for sampling from a normal distribution.

Other rng:
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
[`nv_uniform`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)

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

p <- nv_array(c(0.025, 0.5, 0.975))
nv_qnorm(p)
#> AnvlArray
#>  -1.9600
#>   0.0000
#>   1.9600
#> [ CPUf32{3} ] 
nv_qnorm(p, mean = 1, sd = 2)
#> AnvlArray
#>  -2.9199
#>   1.0000
#>   4.9199
#> [ CPUf32{3} ] 
nv_qnorm(p, lower_tail = FALSE)
#> AnvlArray
#>   1.9600
#>  -0.0000
#>  -1.9600
#> [ CPUf32{3} ] 
nv_qnorm(nv_array(c(-700, -2, -0.1), dtype = "f64"), log_p = TRUE)
#> AnvlArray
#>  -37.2951
#>   -1.1015
#>    1.3096
#> [ CPUf64{3} ] 
# `state` is the updated RNG state, `values` the sample
state <- nv_rng_state(42L)
result <- nv_rnorm(c(2, 3), state)
result$values
#> AnvlArray
#>  -0.0675  1.9457  1.2002
#>   0.9489 -0.5255  0.0008
#> [ CPUf32{2,3} ] 

# `sd` may also be an array of the same shape as the sample
sds <- nv_array(matrix(c(0.01, 0.1, 1, 10, 100, 1000), nrow = 2))
nv_rnorm(c(2, 3), state, sd = sds)$values
#> AnvlArray
#>   -0.0007   1.9457 120.0167
#>    0.0949  -5.2551   0.7665
#> [ CPUf32{2,3} ] 
```
