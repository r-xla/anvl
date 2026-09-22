# Primitive Arc Tangent

Element-wise inverse tangent.

## Usage

``` r
prim_atan(x)
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

- `reverse`

## StableHLO

Lowers to
[`hlo_atan()`](https://r-xla.github.io/stablehlo/reference/hlo_atan.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.atan](https://openxla.org/stablehlo/generated/chlo#chloatan_chloatanop).

## See also

[`nv_atan()`](https://r-xla.github.io/anvl/dev/reference/nv_atan.md),
[`atan()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_atan(x)
#> AnvlArray
#>  -0.7854
#>   0.0000
#>   0.7854
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_atan(1)
#> AnvlArray
#>  0.7854
#> [ CPUf32{} ] 
```
