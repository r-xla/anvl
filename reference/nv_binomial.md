# The Binomial Distribution

Density (`nv_dbinom`), distribution function (`nv_pbinom`), and quantile
function (`nv_qbinom`) for the Binomial distribution with parameters
`size` and `prob`.

## Usage

``` r
nv_dbinom(x, size, prob, log = FALSE)

nv_pbinom(q, size, prob, lower_tail = TRUE, log_p = FALSE)

nv_qbinom(p, size, prob, lower_tail = TRUE, log_p = FALSE)
```

## Arguments

- x, q:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Quantiles at which to evaluate the density (`x`) or the distribution
  function (`q`). These are counts, given as integers or floats; see
  Non-Integer Values for values that are not whole numbers.

- size:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Number of trials, zero or more. Either a scalar, or an array of
  exactly the shape of `x`/`q`/`p`.

- prob:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Probability of success on each trial, in \\\[0, 1\]\\, shaped like
  `size`.

- log, log_p:

  (`logical(1)`)  
  If `TRUE`, the densities/probabilities are given as logarithms. For
  `nv_qbinom` this describes the input `p`.

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
`nv_dbinom()`, `nv_pbinom()`, and `nv_qbinom()` return an
[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) with
the shape of `x`/`q`/`p` and the float data type described under Data
Types.

## Details

The Binomial distribution with `size` \\= n\\ and `prob` \\= p\\ has
probability mass function: \$\$p(x) = \binom{n}{x} p^x (1-p)^{n-x},
\quad x = 0, \ldots, n\$\$ and zero elsewhere.

Invalid arguments give `NaN`, as in base R, but without a warning (that
is `prob` outside \\\[0, 1\]\\, or a `size` that is negative or not a
whole number). `size` is taken as a whole number with the same tolerance
as `x` (see Non-Integer Values).

`nv_dbinom` uses the saddle point expansion of Loader (2000), as base R
does. `nv_pbinom` uses the incomplete beta function methods of DiDonato
& Morris (1992) that base R's
[`stats::pbeta()`](https://rdrr.io/r/stats/Beta.html) uses: a continued
fraction, and an asymptotic expansion for large `size` near the mean. If
the continued fraction has not converged after 1000 iterations, the
result is `NaN`. `nv_qbinom` finds the smallest \\x\\ with \\P(X \le x)
\ge p\\, starting from a Cornish-Fisher approximation as base R does.
Above `2^24` in `f32` and `2^53` in `f64` not every whole number is a
float, and `nv_qbinom` then returns the smallest float \\x\\ with \\P(X
\le x) \ge p\\.

## Non-Integer Values

`x` and `q` are counts, but any value is accepted, and one that is not a
whole number is treated as in base R, without its warnings:

- `nv_dbinom()` takes an `x` within a relative `1e-9` of a whole number,
  that is with \\\|x - \mathrm{round}(x)\| \le 10^{-9} \max(\|x\|, 1)\\,
  as that whole number. Any other `x` has probability zero.

- `nv_pbinom()` rounds `q` down, except that a `q` within `1e-7` below a
  whole number counts as that number.

- A negative `x` or `q` is below the support even where it is within the
  tolerance of zero: such an `x` has probability zero, and such a `q`
  has \\P(X \le q) = 0\\.

Finite quantiles from `nv_qbinom()` are always whole numbers. Since `x`
and `q` are used only as whole numbers, their gradient is zero wherever
a gradient can be taken.

## Gradients

`nv_dbinom` can be differentiated with respect to `prob`, including at
`prob` of 0 and 1. Where the density is zero there, as at `prob = 0` for
`x > 0`, the log-density is `-Inf` and has no derivative: it diverges.
With `log = TRUE` the gradient returned there is zero, which is a
convention of this implementation, not the mathematical derivative.
`nv_pbinom` iterates until convergence in a
[`nv_while()`](https://r-xla.github.io/anvl/reference/nv_while.md) loop,
so [`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)
cannot differentiate it. `nv_qbinom` is a step function, and autodiff
returns a zero gradient for all of its arguments.

## Data Types

`nv_dbinom()`, `nv_pbinom()` and `nv_qbinom()` compute at a float data
type and return it. If `x`/`q`/`p` is a float, they compute at its data
type; an R double settles on the [default
float](https://r-xla.github.io/anvl/reference/default_dtypes.md). An
integer `x`/`q`/`p` (an array or an R integer) is converted to the
widest float among the arrays given for `size` or `prob`, or to the
default float if there is none. Integers beyond \\2^{24}\\ are not all
exact at `f32`, so a large integer may change in that conversion.

An R value for `size` or `prob` takes the computing data type. An array
is promoted to it, so a wider one (e.g. `f64` for an `f32` `x`) is an
error.

## References

Loader C (2000). “Fast and Accurate Computation of Binomial
Probabilities.”
<https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf>.

DiDonato A, Morris A (1992). “Algorithm 708: Significant Digit
Computation of the Incomplete Beta Function Ratios.” *ACM Transactions
on Mathematical Software*, **18**(3), 360–373.
[doi:10.1145/131766.131776](https://doi.org/10.1145/131766.131776) .

Dekker T (1971). “A floating-point technique for extending the available
precision.” *Numerische Mathematik*, **18**, 224–242.
[doi:10.1007/BF01397083](https://doi.org/10.1007/BF01397083) .

## See also

[`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md) for
sampling from a Binomial distribution.

## Examples

``` r
x <- nv_array(c(0, 1, 2, 5, 10))
nv_dbinom(x, size = 10, prob = 0.3)
#> AnvlArray
#>  2.8248e-02
#>  1.2106e-01
#>  2.3347e-01
#>  1.0292e-01
#>  5.9049e-06
#> [ CPUf32{5} ] 
nv_dbinom(x, size = 10, prob = 0.3, log = TRUE)
#> AnvlArray
#>   -3.5667
#>   -2.1115
#>   -1.4547
#>   -2.2738
#>  -12.0397
#> [ CPUf32{5} ] 

nv_pbinom(x, size = 10, prob = 0.3)
#> AnvlArray
#>  0.0282
#>  0.1493
#>  0.3828
#>  0.9527
#>  1.0000
#> [ CPUf32{5} ] 
nv_pbinom(x, size = 10, prob = 0.3, lower_tail = FALSE)
#> AnvlArray
#>  0.9718
#>  0.8507
#>  0.6172
#>  0.0473
#>  0.0000
#> [ CPUf32{5} ] 
nv_pbinom(x, size = 10, prob = 0.3, log_p = TRUE)
#> AnvlArray
#>  -3.5667
#>  -1.9017
#>  -0.9603
#>  -0.0485
#>   0.0000
#> [ CPUf32{5} ] 

p <- nv_array(c(0.025, 0.5, 0.975))
nv_qbinom(p, size = 10, prob = 0.3)
#> AnvlArray
#>  0
#>  3
#>  6
#> [ CPUf32{3} ] 
nv_qbinom(p, size = 10, prob = 0.3, lower_tail = FALSE)
#> AnvlArray
#>  6
#>  3
#>  0
#> [ CPUf32{3} ] 
nv_qbinom(nv_array(c(-700, -2, -0.1), dtype = "f64"), size = 1000, prob = 0.3, log_p = TRUE)
#> AnvlArray
#>    0
#>  284
#>  319
#> [ CPUf64{3} ] 
```
