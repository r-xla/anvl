# Primitive Arc Sine

Element-wise inverse sine.

## Usage

``` r
prim_asin(x)
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
[`hlo_asin()`](https://r-xla.github.io/stablehlo/reference/hlo_asin.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.asin](https://openxla.org/stablehlo/generated/chlo#chloasin_chloasinop).

## See also

[`nv_asin()`](https://r-xla.github.io/anvl/dev/reference/nv_asin.md),
[`asin()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_asin(x)
#> AnvlArray
#>  -1.5708
#>   0.0000
#>   1.5708
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_asin(0.5)
#> AnvlArray
#>  0.5236
#> [ CPUf32{} ] 
```
