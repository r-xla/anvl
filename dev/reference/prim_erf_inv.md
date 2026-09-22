# Primitive Inverse Error Function

Element-wise inverse error function (the inverse of `erf` on `(-1, 1)`).

## Usage

``` r
prim_erf_inv(x)
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
[`hlo_erf_inv()`](https://r-xla.github.io/stablehlo/reference/hlo_erf_inv.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.erf_inv](https://openxla.org/stablehlo/generated/chlo#chloerf_inv_chloerfinvop).

## See also

[`nv_erf_inv()`](https://r-xla.github.io/anvl/dev/reference/nv_erf_inv.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-0.5, 0, 0.5))
prim_erf_inv(x)
#> AnvlArray
#>  -0.4769
#>   0.0000
#>   0.4769
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_erf_inv(0.5)
#> AnvlArray
#>  0.4769
#> [ CPUf32{} ] 
```
