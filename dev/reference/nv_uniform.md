# The Uniform Distribution

Density (`nv_dunif`), distribution function (`nv_punif`), and quantile
function (`nv_qunif`) for the Uniform distribution on the interval from
`min` to `max`.

## Usage

``` r
nv_dunif(x, min = 0, max = 1, log = FALSE)

nv_punif(q, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE)

nv_qunif(p, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE)
```

## Arguments

- x, q:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Quantiles at which to evaluate the density (`x`) or the distribution
  function (`q`).

- min, max:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Lower and upper limits of the distribution. Either scalars, or arrays
  of exactly the same shape as `x`/`q`/`p`, in which case the interval
  varies elementwise and each element of `x`/`q`/`p` is evaluated
  against its own `min`/`max`.

- log, log_p:

  (`logical(1)`)  
  If `TRUE`, the densities/probabilities are given as logarithms. For
  `nv_qunif` this describes the input `p`.

- lower_tail:

  (`logical(1)`)  
  If `TRUE` (default), probabilities are \\P(X \le x)\\; otherwise,
  \\P(X \> x)\\.

- p:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Probabilities at which to evaluate the quantile function. Values
  outside \\\[0, 1\]\\ give `NaN`.

## Value

`nv_dunif()`, `nv_punif()`, and `nv_qunif()` return an
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
with the same shape and data type as `x`/`q`/`p`.

## Details

The Uniform distribution has probability density function: \$\$f(x) =
\frac{1}{b - a}, \quad a \le x \le b\$\$ and zero elsewhere, where \\a\\
is `min` and \\b\\ is `max`. The `min` and `max` are converted to the
data type of `x`/`q`/`p`.

All three are univariate functions evaluated elementwise, returning one
value per element of `x`/`q`/`p`. Non-scalar `min`/`max` therefore give
a separate univariate Uniform per element, *not* a multivariate Uniform
over the hyper-rectangle \\\prod_i \[a_i, b_i\]\\. For that, reduce over
the result: `nv_reduce_prod(nv_dunif(x, min, max))`, or
`nv_reduce_sum(nv_dunif(x, min, max, log = TRUE))` on the log scale.

## See also

[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md)
for sampling from a uniform distribution.

## Examples

``` r
x <- nv_array(c(-0.5, 0, 0.25, 1, 1.5))
nv_dunif(x)
#> AnvlArray
#>  0
#>  1
#>  1
#>  1
#>  0
#> [ CPUf32{5} ] 
nv_dunif(x, min = -1, max = 2)
#> AnvlArray
#>  0.3333
#>  0.3333
#>  0.3333
#>  0.3333
#>  0.3333
#> [ CPUf32{5} ] 
nv_dunif(x, log = TRUE)
#> AnvlArray
#>  -inf
#>     0
#>     0
#>     0
#>  -inf
#> [ CPUf32{5} ] 

# `min`/`max` may vary elementwise, giving one univariate Uniform per
# element rather than a single distribution over a hyper-rectangle
lower <- nv_array(c(-1, -1, 0, 0, 1))
upper <- nv_array(c(0, 1, 1, 2, 2))
nv_dunif(x, min = lower, max = upper)
#> AnvlArray
#>  1.0000
#>  0.5000
#>  1.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{5} ] 

nv_punif(x)
#> AnvlArray
#>  0.0000
#>  0.0000
#>  0.2500
#>  1.0000
#>  1.0000
#> [ CPUf32{5} ] 
nv_punif(x, min = -1, max = 2)
#> AnvlArray
#>  0.1667
#>  0.3333
#>  0.4167
#>  0.6667
#>  0.8333
#> [ CPUf32{5} ] 
nv_punif(x, lower_tail = FALSE)
#> AnvlArray
#>  1.0000
#>  1.0000
#>  0.7500
#>  0.0000
#>  0.0000
#> [ CPUf32{5} ] 
nv_punif(x, log_p = TRUE)
#> AnvlArray
#>     -inf
#>     -inf
#>  -1.3863
#>  -0.0000
#>  -0.0000
#> [ CPUf32{5} ] 

p <- nv_array(c(0.025, 0.5, 0.975))
nv_qunif(p)
#> AnvlArray
#>  0.0250
#>  0.5000
#>  0.9750
#> [ CPUf32{3} ] 
nv_qunif(p, min = -1, max = 2)
#> AnvlArray
#>  -0.9250
#>   0.5000
#>   1.9250
#> [ CPUf32{3} ] 
nv_qunif(p, lower_tail = FALSE)
#> AnvlArray
#>  0.9750
#>  0.5000
#>  0.0250
#> [ CPUf32{3} ] 
nv_qunif(nv_array(c(-700, -2, -0.1), dtype = "f64"), log_p = TRUE)
#> AnvlArray
#>  9.8597e-305
#>   1.3534e-01
#>   9.0484e-01
#> [ CPUf64{3} ] 
```
