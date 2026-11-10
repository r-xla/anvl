# The Poisson Distribution

Density (`nv_dpois`), distribution function (`nv_ppois`), and quantile
function (`nv_qpois`) for the Poisson distribution with mean `lambda`.

## Usage

``` r
nv_dpois(x, lambda, log = FALSE)

nv_ppois(q, lambda, lower_tail = TRUE, log_p = FALSE)

nv_qpois(p, lambda, lower_tail = TRUE, log_p = FALSE)
```

## Arguments

- x, q:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Quantiles at which to evaluate the density (`x`) or the distribution
  function (`q`). These are counts, given as integers or floats; see
  Non-Integer Values for values that are not whole numbers.

- lambda:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Mean of the distribution. Scalars broadcast; otherwise `lambda` and
  `x`/`q`/`p` must have the same shape.

- log, log_p:

  (`logical(1)`)  
  If `TRUE`, the densities/probabilities are given as logarithms. For
  `nv_qpois` this describes the input `p`.

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
`nv_dpois()`, `nv_ppois()`, and `nv_qpois()` return an
[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) with
the float data type described under Data Types and the shape after
scalar broadcasting with `lambda`.

## Details

The Poisson distribution has probability mass function: \$\$p(x) =
\frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0, 1, 2, \ldots\$\$ where
\\\lambda\\ is `lambda`.

Invalid arguments give `NaN`, as in base R, but without a warning (that
is a negative `lambda`). Zero and infinite `lambda` follow base R's
endpoint conventions:

- `lambda = 0`: the point mass at zero.

- `lambda = Inf`: densities are zero. The lower-tail probability is zero
  at finite `q` and one at `q = Inf`. Quantiles are `NaN`.

`nv_dpois` uses the saddle point expansion of Loader (2000), as base R
does. `nv_ppois` uses the incomplete gamma function methods of base R's
[`stats::pgamma()`](https://rdrr.io/r/stats/GammaDist.html): a series in
the tails, and an asymptotic expansion where `q` and `lambda` are large
and close. `nv_qpois` finds the smallest \\x\\ with \\P(X \le x) \ge
p\\, starting from a Cornish-Fisher approximation as base R does. Above
`2^24` in `f32` and `2^53` in `f64` not every whole number is a float,
and `nv_qpois` then returns the smallest float \\x\\ with \\P(X \le x)
\ge p\\.

## Non-Integer Values

`x` and `q` are counts, but any value is accepted, and one that is not a
whole number is treated as in base R, without its warnings:

- `nv_dpois()` takes an `x` within a relative `1e-9` of a whole number,
  that is with \\\|x - \mathrm{round}(x)\| \le 10^{-9} \max(\|x\|, 1)\\,
  as that whole number. Any other `x` has probability zero.

- `nv_ppois()` rounds `q` down, except that a `q` within `1e-7` below a
  whole number counts as that number.

- A negative `x` or `q` is below the support even where it is within the
  tolerance of zero: such an `x` has probability zero, and such a `q`
  has \\P(X \le q) = 0\\.

Finite quantiles from `nv_qpois()` are always whole numbers. Since `x`
and `q` are used only as whole numbers, their gradient is zero wherever
a gradient can be taken.

## Gradients

`nv_dpois` and `nv_ppois` can be differentiated with respect to
`lambda`, to any order. At `lambda = 0`, where only a right derivative
exists, the implementation returns it for `nv_dpois(0, lambda)` and, by
convention, zero for the other probabilities at `lambda = 0`. `nv_qpois`
is a step function, and autodiff returns a zero gradient for all of its
arguments.

## Data Types

`nv_dpois()`, `nv_ppois()` and `nv_qpois()` compute at a float data type
and return it. If `x`/`q`/`p` is a float, they compute at its data type;
an R double settles on the [default
float](https://r-xla.github.io/anvl/reference/default_dtypes.md). An
integer `x`/`q`/`p` (an array or an R integer) is converted to the
widest float among the arrays given for `lambda`, or to the default
float if there is none. Integers beyond \\2^{24}\\ are not all exact at
`f32`, so a large integer may change in that conversion.

An R value for `lambda` takes the computing data type. An array is
promoted to it, so a wider one (e.g. `f64` for an `f32` `x`) is an
error.

## References

Loader C (2000). “Fast and Accurate Computation of Binomial
Probabilities.”
<https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf>.

## Examples

``` r
x <- nv_array(c(0, 1, 2, 5, 10))
nv_dpois(x, lambda = 4)
#> AnvlArray
#>  0.0183
#>  0.0733
#>  0.1465
#>  0.1563
#>  0.0053
#> [ CPUf32{5} ] 
nv_dpois(x, lambda = 4, log = TRUE)
#> AnvlArray
#>  -4.0000
#>  -2.6137
#>  -1.9206
#>  -1.8560
#>  -5.2415
#> [ CPUf32{5} ] 

nv_ppois(x, lambda = 4)
#> AnvlArray
#>  0.0183
#>  0.0916
#>  0.2381
#>  0.7851
#>  0.9972
#> [ CPUf32{5} ] 
nv_ppois(x, lambda = 4, lower_tail = FALSE)
#> AnvlArray
#>  0.9817
#>  0.9084
#>  0.7619
#>  0.2149
#>  0.0028
#> [ CPUf32{5} ] 
nv_ppois(x, lambda = 4, log_p = TRUE)
#> AnvlArray
#>  -4.0000
#>  -2.3906
#>  -1.4351
#>  -0.2419
#>  -0.0028
#> [ CPUf32{5} ] 

p <- nv_array(c(0.025, 0.5, 0.975))
nv_qpois(p, lambda = 4)
#> AnvlArray
#>  1
#>  4
#>  8
#> [ CPUf32{3} ] 
nv_qpois(p, lambda = 4, lower_tail = FALSE)
#> AnvlArray
#>  8
#>  4
#>  1
#> [ CPUf32{3} ] 
nv_qpois(
  nv_array(c(-700, -2, -0.1), dtype = "f64"),
  lambda = 1000,
  log_p = TRUE
)
#> AnvlArray
#>    89
#>   965
#>  1042
#> [ CPUf64{3} ] 
```
