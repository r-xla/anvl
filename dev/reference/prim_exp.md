# Primitive Exponential

Element-wise exponential.

## Usage

``` r
prim_exp(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any float data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_exponential()`](https://r-xla.github.io/stablehlo/reference/hlo_exponential.html),
specified under
[exponential](https://openxla.org/stablehlo/spec#exponential).

## See also

[`nv_exp()`](https://r-xla.github.io/anvl/dev/reference/nv_exp.md),
[`exp()`](https://rdrr.io/r/base/Log.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, 1, 2))
prim_exp(x)
#> AnvlArray
#>  1.0000
#>  2.7183
#>  7.3891
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_exp(1)
#> AnvlArray
#>  2.7183
#> [ CPUf32{} ] 
```
