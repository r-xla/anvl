# Sample from a Uniform Distribution

Samples from a uniform distribution in the open interval `(min, max)`.

## Usage

``` r
nv_runif(shape, initial_state, dtype = "f32", min = 0, max = 1)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  RNG state (`ui64[2]`).

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

- min, max:

  (`numeric(1)`)  
  Lower and upper bound.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
List of two elements: the updated RNG state and the sampled values.

## See also

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/reference/nv_rbinom.md),
[`nv_rng_state()`](https://r-xla.github.io/anvl/reference/nv_rng_state.md),
[`nv_sample()`](https://r-xla.github.io/anvl/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/reference/nv_sample_int.md)

## Examples

``` r
state <- nv_rng_state(42L)
result <- nv_runif(c(2, 3), state)
result[[2]]
#> AnvlArray
#>  0.8690 0.1506 0.5203
#>  0.3103 0.9928 0.1065
#> [ CPUf32{2,3} ] 
```
