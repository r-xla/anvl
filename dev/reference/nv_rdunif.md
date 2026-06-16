# Sample from a Discrete Uniform Distribution

Samples integers from `1` to `n` with equal probability (with
replacement), analogous to R's `sample.int(n, size, replace = TRUE)`.

## Usage

``` r
nv_rdunif(shape, initial_state, n, dtype = "i32")
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  RNG state (`ui64[2]`).

- n:

  (`integer(1)`)  
  Number of categories (samples integers `1` to `n`).

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
List of two elements: the updated RNG state and the sampled integers.

## See also

Other rng:
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/dev/reference/nv_rng_state.md),
[`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_rnorm.md),
[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md)

## Examples

``` r
state <- nv_rng_state(42L)
# Roll 6 dice
result <- nv_rdunif(6, state, n = 6L)
result[[2]]
#> AnvlArray
#>  3
#>  5
#>  1
#>  4
#>  1
#>  1
#> [ CPUi32{6} ] 
```
