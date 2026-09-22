# Primitive Error Function

Element-wise error function
`erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt`.

## Usage

``` r
prim_erf(x)
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
[`hlo_erf()`](https://r-xla.github.io/stablehlo/reference/hlo_erf.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.erf](https://openxla.org/stablehlo/generated/chlo#chloerf_chloerfop).

## See also

[`nv_erf()`](https://r-xla.github.io/anvl/dev/reference/nv_erf.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_erf(x)
#> AnvlArray
#>  -0.8427
#>   0.0000
#>   0.8427
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_erf(1)
#> AnvlArray
#>  0.8427
#> [ CPUf32{} ] 
```
