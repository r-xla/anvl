# Sample from a Binomial Distribution

Samples from a binomial distribution with \\n\\ trials and success
probability \\p\\. When `size = 1` (the default), this is a Bernoulli
distribution.

## Usage

``` r
nv_rbinom(shape, size = 1L, prob = 0.5, dtype = NULL)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the result.

- size:

  (`integer(1)`)  
  Number of trials. `0` gives a sample of zeros, like
  [`rbinom()`](https://rdrr.io/r/stats/Binomial.html).

- prob:

  (`numeric(1)`)  
  Probability of success on each trial.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/xlamisc/reference/DataType.html))  
  Numeric type of the sample. `NULL` (default) uses the [default integer
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md). The
  number of successes are converted to it, so it must hold `size`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
The sample, of shape `shape` and data type `dtype`.

## Random State

The draws come from the active random state: the global random state
(see
[`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md)),
or the random state that
[`with_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md)
or
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md)
make the default for a scope. They advance it in place.

## See also

Other rng:
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md),
[`nv_normal`](https://r-xla.github.io/anvl/reference/nv_normal.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md),
[`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md),
[`nv_uniform`](https://r-xla.github.io/anvl/reference/nv_uniform.md)

## Examples

``` r
# Bernoulli samples
with_nv_seed(42L, nv_rbinom(c(2, 3)))
#> AnvlArray
#>  1 0 0
#>  1 1 0
#> [ CPUi32{2,3} ] 
```
