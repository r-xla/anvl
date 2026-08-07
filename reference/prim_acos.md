# Primitive Arc Cosine

Element-wise inverse cosine.

## Usage

``` r
prim_acos(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_acos()`](https://r-xla.github.io/stablehlo/reference/hlo_acos.html).

## See also

[`nv_acos()`](https://r-xla.github.io/anvl/reference/nv_acos.md),
[`acos()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_acos(x)
#> AnvlArray
#>  3.1416
#>  1.5708
#>  0.0000
#> [ CPUf32{3} ] 
```
