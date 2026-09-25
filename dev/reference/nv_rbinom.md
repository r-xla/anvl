# Sample from a Binomial Distribution

Samples from a binomial distribution with \\n\\ trials and success
probability \\p\\. When `size = 1` (the default), this is a Bernoulli
distribution.

## Usage

``` r
nv_rbinom(shape, state, size = 1L, prob = 0.5, dtype = NULL)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the result.

- state:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  RNG state: a 1-D array of two `ui64` elements, as
  [`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md)
  returns. The data type and length are fixed by the generator, not by
  the default data types, and the returned `state` has them too.

- size:

  (`integer(1)`)  
  Number of trials.

- prob:

  (`numeric(1)`)  
  Probability of success on each trial.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Numeric type of the sample. `NULL` (default) uses the [default integer
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).
  The number of successes are converted to it.

## Value

(named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `state`, the updated RNG state, and `values`, the sample of
shape `shape` and data type `dtype`.

## See also

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
[`nv_uniform`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)

## Examples

``` r
# Bernoulli samples; `state` is the updated RNG state
state <- nv_rng_state(42L)
result <- nv_rbinom(c(2, 3), state)
result$values
#> AnvlArray
#>  0 1 1
#>  0 0 1
#> [ CPUi32{2,3} ] 
```
