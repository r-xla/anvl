# Primitive RNG Bit Generator

Generates pseudo-random numbers using the specified algorithm and
returns the updated RNG state together with the generated values.

## Usage

``` r
prim_rng_bit_generator(
  initial_state,
  rng_algorithm = "THREE_FRY",
  dtype,
  shape
)
```

## Arguments

- initial_state:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  RNG state: a 1-D array of the `ui64` data type. Its length depends on
  `rng_algorithm` – exactly 2 for `"THREE_FRY"`, 2 or 3 for `"PHILOX"`,
  and whatever the implementation wants for `"DEFAULT"`.

- rng_algorithm:

  (`character(1)`)  
  One of `"THREE_FRY"` (default), `"PHILOX"` or `"DEFAULT"`, the last
  leaving the choice to the implementation.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type of the generated random values. Can be any numeric data
  type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the result.

## Value

(named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `state`, the updated RNG state with `initial_state`'s data type
and shape, and `values`, the random values with the given `dtype` and
`shape`.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_rng_bit_generator()`](https://r-xla.github.io/stablehlo/reference/hlo_rng_bit_generator.html),
specified under
[rng_bit_generator](https://openxla.org/stablehlo/spec#rng_bit_generator).

## See also

[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md),
[`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md)

## Examples

``` r
# THREE_FRY, the default, takes a two-element state
state <- nv_array(c(0L, 0L), dtype = "ui64")
prim_rng_bit_generator(state, dtype = "f32", shape = c(3, 2))
#> $state
#> AnvlArray
#>  0
#>  3
#> [ CPUui64{2} ] 
#> 
#> $values
#> AnvlArray
#>  1.7973e+09 2.5791e+09
#>  1.3515e+09 3.2358e+09
#>  1.6886e+09 4.2293e+09
#> [ CPUf32{3,2} ] 
#> 

# the updated state feeds the next draw, so the two differ
out <- prim_rng_bit_generator(state, dtype = "f32", shape = 3L)
prim_rng_bit_generator(out$state, dtype = "f32", shape = 3L)
#> $state
#> AnvlArray
#>  0
#>  4
#> [ CPUui64{2} ] 
#> 
#> $values
#> AnvlArray
#>  1.6886e+09
#>  4.2293e+09
#>  3.0983e+09
#> [ CPUf32{3} ] 
#> 

# PHILOX also accepts a three-element state
prim_rng_bit_generator(
  nv_array(c(0L, 0L, 0L), dtype = "ui64"), "PHILOX",
  dtype = "i32", shape = 4L
)
#> $state
#> AnvlArray
#>  0
#>  1
#>  0
#> [ CPUui64{3} ] 
#> 
#> $values
#> AnvlArray
#>  1.7139e+09
#>  3.7818e+09
#>  3.1599e+09
#>  2.6005e+09
#> [ CPUi32{4} ] 
#> 
```
