# The Exponential Distribution

Density (`nv_dexp`), distribution function (`nv_pexp`), and quantile
function (`nv_qexp`) for the Exponential distribution with rate `rate`
(i.e., mean `1 / rate`).

## Usage

``` r
nv_dexp(x, rate = 1, log = FALSE)

nv_pexp(q, rate = 1, lower_tail = TRUE, log_p = FALSE)

nv_qexp(p, rate = 1, lower_tail = TRUE, log_p = FALSE)
```

## Arguments

- x, q:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Quantiles at which to evaluate the density (`x`) or the distribution
  function (`q`).

- rate:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Rate of the distribution. Scalars broadcast; otherwise `rate` and
  `x`/`q`/`p` must have the same shape. Negative rates (including `-Inf`
  and negative zero) give `NaN`.

- log, log_p:

  (`logical(1)`)  
  If `TRUE`, the densities/probabilities are given as logarithms. For
  `nv_qexp` this describes the input `p`.

- lower_tail:

  (`logical(1)`)  
  If `TRUE` (default), probabilities are \\P(X \le x)\\; otherwise,
  \\P(X \> x)\\.

- p:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Probabilities at which to evaluate the quantile function. Values
  outside \\\[0, 1\]\\ give `NaN`. With `log_p = TRUE`, supply
  log-probabilities in \\\[-\infty, 0\]\\ instead.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
`nv_dexp()`, `nv_pexp()`, and `nv_qexp()` return an
[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) with
the data type of `x`/`q`/`p` and the shape after scalar broadcasting
with `rate`.

## Details

The Exponential distribution has probability density function: \$\$f(x)
= \lambda e^{-\lambda x}, \quad x \ge 0\$\$ and zero elsewhere, where
\\\lambda\\ is `rate`.

For positive finite rates, the mean is `1 / rate`. Zero and infinite
rates follow base R's endpoint conventions:

- `rate = 0`: the density and lower-tail probability are zero at finite
  quantiles, but `NaN` at infinity. Quantiles are infinite except at
  lower-tail probability zero, where the result is zero.

- `rate = Inf`: densities are `NaN`. The lower-tail probability is zero
  at or below zero and one above zero. Quantiles are zero except at
  lower-tail probability one, where the result is `NaN`.

## Data Types

`nv_dexp()`, `nv_pexp()` and `nv_qexp()` compute at the data type of
`x`/`q`/`p`, which must be float; an R value settles on the [default
float](https://r-xla.github.io/anvl/reference/default_dtypes.md). An R
value for `rate` takes this data type. An array is promoted to it, so a
wider one (e.g. `f64` for an `f32` `x`) is an error.

## Examples

``` r
x <- nv_array(c(-1, 0, 0.5, 1, 2))
nv_dexp(x)
#> AnvlArray
#>  0.0000
#>  1.0000
#>  0.6065
#>  0.3679
#>  0.1353
#> [ CPUf32{5} ] 
nv_dexp(x, rate = 2)
#> AnvlArray
#>  0.0000
#>  2.0000
#>  0.7358
#>  0.2707
#>  0.0366
#> [ CPUf32{5} ] 
nv_dexp(x, log = TRUE)
#> AnvlArray
#>     -inf
#>   0.0000
#>  -0.5000
#>  -1.0000
#>  -2.0000
#> [ CPUf32{5} ] 

nv_pexp(x)
#> AnvlArray
#>  0.0000
#>  0.0000
#>  0.3935
#>  0.6321
#>  0.8647
#> [ CPUf32{5} ] 
nv_pexp(x, rate = 2)
#> AnvlArray
#>  0.0000
#>  0.0000
#>  0.6321
#>  0.8647
#>  0.9817
#> [ CPUf32{5} ] 
nv_pexp(x, lower_tail = FALSE)
#> AnvlArray
#>  1.0000
#>  1.0000
#>  0.6065
#>  0.3679
#>  0.1353
#> [ CPUf32{5} ] 
nv_pexp(x, log_p = TRUE)
#> AnvlArray
#>     -inf
#>     -inf
#>  -0.9328
#>  -0.4587
#>  -0.1454
#> [ CPUf32{5} ] 

p <- nv_array(c(0.025, 0.5, 0.975))
nv_qexp(p)
#> AnvlArray
#>  0.0253
#>  0.6931
#>  3.6889
#> [ CPUf32{3} ] 
nv_qexp(p, rate = 2)
#> AnvlArray
#>  0.0127
#>  0.3466
#>  1.8444
#> [ CPUf32{3} ] 
nv_qexp(p, lower_tail = FALSE)
#> AnvlArray
#>  3.6889
#>  0.6931
#>  0.0253
#> [ CPUf32{3} ] 
nv_qexp(nv_array(c(-700, -2, -0.1), dtype = "f64"), log_p = TRUE)
#> AnvlArray
#>  9.8597e-305
#>   1.4541e-01
#>   2.3522e+00
#> [ CPUf64{3} ] 
```
