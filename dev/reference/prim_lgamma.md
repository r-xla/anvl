# Primitive Log-Gamma

Element-wise natural logarithm of the absolute value of the gamma
function.

## Usage

``` r
prim_lgamma(x)
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
[`hlo_lgamma()`](https://r-xla.github.io/stablehlo/reference/hlo_lgamma.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.lgamma](https://openxla.org/stablehlo/generated/chlo#chlolgamma_chlolgammaop).

## See also

[`nv_lgamma()`](https://r-xla.github.io/anvl/dev/reference/nv_lgamma.md),
[`lgamma()`](https://rdrr.io/r/base/Special.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0.5, 1, 2, 5))
prim_lgamma(x)
#> AnvlArray
#>  5.7236e-01
#>  4.7684e-07
#>  0.0000e+00
#>  3.1781e+00
#> [ CPUf32{4} ] 

# an R value materializes at its default data type
prim_lgamma(2)
#> AnvlArray
#>  0
#> [ CPUf32{} ] 
```
