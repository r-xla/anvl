# Sample from a Binomial Distribution

Samples from a binomial distribution with \\n\\ trials and success
probability \\p\\. When `size = 1` (the default), this is a Bernoulli
distribution.

## Usage

``` r
nv_rbinom(shape, initial_state, size = 1L, prob = 0.5, dtype = "i32")
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  RNG state (`ui64[2]`).

- size:

  (`integer(1)`)  
  Number of trials.

- prob:

  (`numeric(1)`)  
  Probability of success on each trial.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
List of two elements: the updated RNG state and the sampled values.

## See also

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md),
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)

## Examples

``` r
state <- nv_rng_state(42L)
# Bernoulli samples
result <- nv_rbinom(c(2, 3), state)
result[[2]]
#> AnvlArray
#>  0 0 1
#>  0 1 1
#> [ CPUi32{2,3} ] 
```
