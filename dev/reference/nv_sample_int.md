# Sample Integers

Samples integers from `1` to `n` with equal probability and with
replacement, analogous to R's
[`sample.int()`](https://rdrr.io/r/base/sample.html).

To sample from a population other than `1:n`, use
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md).

## Usage

``` r
nv_sample_int(shape, state, n, dtype = NULL)
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

- n:

  (`integer(1)`)  
  Size of the population, i.e. the integers `1` to `n` are sampled.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Numeric type of the sampled integers. The sampled values are converted
  to it. `NULL` (default) uses the [default integer
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

(named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `state`, the updated RNG state, and `values`, the sampled
integers of shape `shape` and data type `dtype`.

## See also

[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md)
to sample from an arbitrary population.

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
[`nv_uniform`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)

## Examples

``` r
# roll six dice; `state` is the updated RNG state
state <- nv_rng_state(42L)
result <- nv_sample_int(6, state, 6L)
result$values
#> AnvlArray
#>  4
#>  6
#>  2
#>  5
#>  1
#>  2
#> [ CPUi32{6} ] 
```
