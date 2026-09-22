# Primitive Arc Cosine

Element-wise inverse cosine.

## Usage

``` r
prim_acos(x)
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
[`hlo_acos()`](https://r-xla.github.io/stablehlo/reference/hlo_acos.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.acos](https://openxla.org/stablehlo/generated/chlo#chloacos_chloacosop).

## See also

[`nv_acos()`](https://r-xla.github.io/anvl/dev/reference/nv_acos.md),
[`acos()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_acos(x)
#> AnvlArray
#>  3.1416
#>  1.5708
#>  0.0000
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_acos(0.5)
#> AnvlArray
#>  1.0472
#> [ CPUf32{} ] 
```
