# Primitive Inverse Hyperbolic Tangent

Element-wise inverse hyperbolic tangent.

## Usage

``` r
prim_atanh(operand)
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

- `reverse`

## StableHLO

Lowers to
[`hlo_atanh()`](https://r-xla.github.io/stablehlo/reference/hlo_atanh.html).

## See also

[`nv_atanh()`](https://r-xla.github.io/anvl/dev/reference/nv_atanh.md),
[`atanh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
x <- nv_array(c(-0.5, 0, 0.5))
prim_atanh(x)
#> AnvlArray
#>  -0.5493
#>   0.0000
#>   0.5493
#> [ CPUf32{3} ] 
```
