# Primitive Inverse Hyperbolic Cosine

Element-wise inverse hyperbolic cosine.

## Usage

``` r
prim_acosh(operand)
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
[`hlo_acosh()`](https://r-xla.github.io/stablehlo/reference/hlo_acosh.html).

## See also

[`nv_acosh()`](https://r-xla.github.io/anvl/dev/reference/nv_acosh.md),
[`acosh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
x <- nv_array(c(1, 2, 10))
prim_acosh(x)
#> AnvlArray
#>  0.0000
#>  1.3170
#>  2.9932
#> [ CPUf32{3} ] 
```
