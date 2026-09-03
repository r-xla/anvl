# Primitive Arc Tangent

Element-wise inverse tangent.

## Usage

``` r
prim_atan(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_atan()`](https://r-xla.github.io/stablehlo/reference/hlo_atan.html).

## See also

[`nv_atan()`](https://r-xla.github.io/anvl/dev/reference/nv_atan.md),
[`atan()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_atan(x)
#> AnvlArray
#>  -0.7854
#>   0.0000
#>   0.7854
#> [ CPUf32{3} ] 
```
