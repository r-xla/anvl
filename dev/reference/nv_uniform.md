# The Uniform Distribution

Density (`nv_dunif`), distribution function (`nv_punif`), quantile
function (`nv_qunif`), and random generation (`nv_runif`) for the
Uniform distribution on the interval from `min` to `max`.

## Usage

``` r
nv_dunif(x, min = 0, max = 1, log = FALSE)

nv_punif(q, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE)

nv_qunif(p, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE)

nv_runif(shape, state, min = 0, max = 1, dtype = NULL)
```

## Arguments

- x, q:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Quantiles at which to evaluate the density (`x`) or the distribution
  function (`q`).

- min, max:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Lower and upper limits of the distribution. Either scalars, or arrays
  of exactly the same shape as `x`/`q`/`p` (or the sample, for
  `nv_runif`), in which case the interval varies elementwise and each
  element of `x`/`q`/`p` is evaluated against, or each draw made from,
  its own `min`/`max`.

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
  Floating point data type of the sample. The default (`NULL`) takes it
  from `min` and `max`, and uses the [default float
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  where both are R values.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) \|
named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
`nv_dunif()`, `nv_punif()`, and `nv_qunif()` return an
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
with the same shape and data type as `x`/`q`/`p`.

`nv_runif()` returns a named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md):
`state`, the updated RNG state with the input `state`'s data type and
shape, and `values`, the sample of shape `shape` and the data type
described under `dtype`.

## Details

The Uniform distribution has probability density function: \$\$f(x) =
\frac{1}{b - a}, \quad a \le x \le b\$\$ and zero elsewhere, where \\a\\
is `min` and \\b\\ is `max`. For `nv_dunif`, `nv_punif`, and `nv_qunif`,
the `min` and `max` are converted to the data type of `x`/`q`/`p`.

All four are univariate functions evaluated elementwise, returning one
value per element of `x`/`q`/`p` (or of the sample). Non-scalar
`min`/`max` therefore give a separate univariate Uniform per element,
*not* a multivariate Uniform over the hyper-rectangle \\\prod_i \[a_i,
b_i\]\\. For that, reduce over the result:
`nv_prod(nv_dunif(x, min, max))`, or
`nv_sum(nv_dunif(x, min, max, log = TRUE))` on the log scale.

## Random generation

`nv_runif` samples from the open interval \\(a, b)\\.

`min` and `max` are
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md), so
they may vary across the sample: they are applied to the draws after
they have been reshaped to `shape`, and so may either be scalars or have
exactly that shape. As in base R's
[`runif()`](https://rdrr.io/r/stats/Uniform.html), an element whose
`min` equals its `max` is that value, and one whose `min` or `max` is
not finite, or whose `max` is less than its `min`, is `NaN`. The RNG
state is advanced regardless.

## See also

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)

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

# `state` is the updated RNG state, `values` the sample
state <- nv_rng_state(42L)
result <- nv_runif(c(2, 3), state)
result$values
#> AnvlArray
#>  0.8690 0.1506 0.5203
#>  0.3103 0.9928 0.1065
#> [ CPUf32{2,3} ] 

# `min`/`max` may also be arrays of the same shape as the sample
lower <- nv_array(matrix(c(0, 10, 20, 30, 40, 50), nrow = 2))
nv_runif(c(2, 3), state, min = lower, max = lower + 1)$values
#> AnvlArray
#>   0.8690 20.1506 40.5203
#>  10.3103 30.9928 50.1065
#> [ CPUf32{2,3} ] 
```
