# Sample from a Population

Samples elements of a 1-D array with equal probability and with
replacement, analogous to R's
[`sample()`](https://rdrr.io/r/base/sample.html).

Unlike R's [`sample()`](https://rdrr.io/r/base/sample.html), `x` is
always the population itself: sampling the integers `1` to `n` is
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)
and never an overload of `x`.

## Usage

``` r
nv_sample(shape, state, x)
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

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The population vector to sample from. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

(named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `state`, the updated RNG state, and `values`, the sample of
shape `shape` and `x`'s data type.

## See also

[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)
to sample the integers `1` to `n`.

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
[`nv_uniform`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)

## Examples

``` r
# the sample takes the population's data type
state <- nv_rng_state(42L)
pop <- nv_array(c(10, 20, 30))
result <- nv_sample(5, state, pop)
result$values
#> AnvlArray
#>  20
#>  30
#>  10
#>  30
#>  10
#> [ CPUf32{5} ] 
```
