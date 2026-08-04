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
nv_sample(shape, initial_state, x)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  RNG state (`ui64[2]`).

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The population to sample from, a 1-D array.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
List of two elements: the updated RNG state and the sampled values, of
shape `shape` and with the data type of `x`.

## See also

[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)
to sample the integers `1` to `n`.

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md)

## Examples

``` r
state <- nv_rng_state(42L)
pop <- nv_array(c(10, 20, 30))
result <- nv_sample(5, state, pop)
result[[2]]
#> AnvlArray
#>  20
#>  30
#>  10
#>  30
#>  10
#> [ CPUf32{5} ] 
```
