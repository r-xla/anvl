# Primitive Inverse Hyperbolic Tangent

Element-wise inverse hyperbolic tangent.

## Usage

``` r
prim_atanh(x)
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
[`hlo_atanh()`](https://r-xla.github.io/stablehlo/reference/hlo_atanh.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.atanh](https://openxla.org/stablehlo/generated/chlo#chloatanh_chloatanhop).

## See also

[`nv_atanh()`](https://r-xla.github.io/anvl/dev/reference/nv_atanh.md),
[`atanh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-0.5, 0, 0.5))
prim_atanh(x)
#> AnvlArray
#>  -0.5493
#>   0.0000
#>   0.5493
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_atanh(0.5)
#> AnvlArray
#>  0.5493
#> [ CPUf32{} ] 
```
