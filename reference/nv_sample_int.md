# Sample Integers

Samples integers from `1` to `n` with equal probability and with
replacement, analogous to R's
[`sample.int()`](https://rdrr.io/r/base/sample.html).

To sample from a population other than `1:n`, use
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md).

## Usage

``` r
nv_sample_int(shape, n, dtype = NULL)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the result.

- n:

  (`integer(1)`)  
  Size of the population, i.e. the integers `1` to `n` are sampled.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/xlamisc/reference/DataType.html))  
  Numeric type of the sampled integers. The sampled values are converted
  to it, so it must hold `n`. `NULL` (default) uses the [default integer
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
The sampled integers, of shape `shape` and data type `dtype`.

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

[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md) to
sample from an arbitrary population.

Other rng:
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md),
[`nv_normal`](https://r-xla.github.io/anvl/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md),
[`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md),
[`nv_uniform`](https://r-xla.github.io/anvl/reference/nv_uniform.md)

## Examples

``` r
# roll six dice
with_nv_seed(42L, nv_sample_int(6, 6L))
#> AnvlArray
#>  2
#>  3
#>  6
#>  3
#>  6
#>  6
#> [ CPUi32{6} ] 
```
