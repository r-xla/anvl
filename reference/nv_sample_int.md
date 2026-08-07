# Sample Integers

Samples integers from `1` to `n` with equal probability and with
replacement, analogous to R's
[`sample.int()`](https://rdrr.io/r/base/sample.html).

To sample from a population other than `1:n`, use
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md).

## Usage

``` r
nv_sample_int(shape, initial_state, n, dtype = "i32")
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  RNG state (`ui64[2]`).

- n:

  (`integer(1)`)  
  Size of the population, i.e. the integers `1` to `n` are sampled.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type of the sampled integers.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
List of two elements: the updated RNG state and the sampled integers, of
shape `shape`.

## See also

[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md) to
sample from an arbitrary population.

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md),
[`nv_runif()`](https://r-xla.github.io/anvl/reference/nv_runif.md),
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md)

## Examples

``` r
state <- nv_rng_state(42L)
# Roll 6 dice
result <- nv_sample_int(6, state, 6L)
result[[2]]
#> AnvlArray
#>  4
#>  6
#>  2
#>  5
#>  1
#>  2
#> [ CPUi32{6} ] 
```
