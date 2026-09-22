# Primitive Complementary Error Function

Element-wise complementary error function `erfc(x) = 1 - erf(x)`.

## Usage

``` r
prim_erfc(x)
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
[`hlo_erfc()`](https://r-xla.github.io/stablehlo/reference/hlo_erfc.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.erfc](https://openxla.org/stablehlo/generated/chlo#chloerfc_chloerfcop).

## See also

[`nv_erfc()`](https://r-xla.github.io/anvl/dev/reference/nv_erfc.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_erfc(x)
#> AnvlArray
#>  1.8427
#>  1.0000
#>  0.1573
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_erfc(1)
#> AnvlArray
#>  0.1573
#> [ CPUf32{} ] 
```
