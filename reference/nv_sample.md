# Sample from a Population

Samples elements of a 1-D array with equal probability and with
replacement, analogous to R's
[`sample()`](https://rdrr.io/r/base/sample.html).

Unlike R's [`sample()`](https://rdrr.io/r/base/sample.html), `x` is
always the population itself: sampling the integers `1` to `n` is
[`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md)
and never an overload of `x`.

## Usage

``` r
nv_sample(shape, x)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the result.

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  The population vector to sample from. Must not be empty. An R value
  materializes at its [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
The sample, of shape `shape` and `x`'s data type.

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

[`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md)
to sample the integers `1` to `n`.

Other rng:
[`local_nv_rng_state()`](https://r-xla.github.io/anvl/reference/local_nv_rng_state.md),
[`nv_normal`](https://r-xla.github.io/anvl/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md),
[`nv_set_seed()`](https://r-xla.github.io/anvl/reference/nv_set_seed.md),
[`nv_uniform`](https://r-xla.github.io/anvl/reference/nv_uniform.md)

## Examples

``` r
# the sample takes the population's data type
with_nv_seed(42L, nv_sample(5, nv_array(c(10, 20, 30))))
#> AnvlArray
#>  10
#>  20
#>  30
#>  20
#>  30
#> [ CPUf32{5} ] 
```
