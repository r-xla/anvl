# Primitive Tangent

Element-wise tangent.

## Usage

``` r
prim_tan(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_tan()`](https://r-xla.github.io/stablehlo/reference/hlo_tan.html).

## See also

[`nv_tan()`](https://r-xla.github.io/anvl/dev/reference/nv_tan.md),
[`tan()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
x <- nv_array(c(0, 0.5, 1))
prim_tan(x)
#> AnvlArray
#>  0.0000
#>  0.5463
#>  1.5574
#> [ CPUf32{3} ] 
```
