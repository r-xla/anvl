# Primitive Hyperbolic Tangent

Element-wise hyperbolic tangent.

## Usage

``` r
prim_tanh(x)
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
[`hlo_tanh()`](https://r-xla.github.io/stablehlo/reference/hlo_tanh.html),
specified under [tanh](https://openxla.org/stablehlo/spec#tanh).

## See also

[`nv_tanh()`](https://r-xla.github.io/anvl/dev/reference/nv_tanh.md),
[`tanh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_tanh(x)
#> AnvlArray
#>  -0.7616
#>   0.0000
#>   0.7616
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_tanh(1)
#> AnvlArray
#>  0.7616
#> [ CPUf32{} ] 
```
