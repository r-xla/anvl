# Primitive Exponential Minus One

Element-wise exp(x) - 1, more accurate for small x.

## Usage

``` r
prim_expm1(x)
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
[`hlo_exponential_minus_one()`](https://r-xla.github.io/stablehlo/reference/hlo_exponential_minus_one.html),
specified under
[exponential_minus_one](https://openxla.org/stablehlo/spec#exponential_minus_one).

## See also

[`nv_expm1()`](https://r-xla.github.io/anvl/dev/reference/nv_expm1.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, 0.001, 1))
prim_expm1(x)
#> AnvlArray
#>  0.0000
#>  0.0010
#>  1.7183
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_expm1(0.001)
#> AnvlArray
#>  0.0010
#> [ CPUf32{} ] 
```
