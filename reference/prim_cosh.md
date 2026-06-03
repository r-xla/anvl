# Primitive Hyperbolic Cosine

Element-wise hyperbolic cosine.

## Usage

``` r
prim_cosh(operand)
```

## Arguments

- operand:

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
[`stablehlo::hlo_cosh()`](https://r-xla.github.io/stablehlo/reference/hlo_cosh.html).

## See also

[`nv_cosh()`](https://r-xla.github.io/anvl/reference/nv_cosh.md),
[`cosh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_cosh(x)
#> AnvlArray
#>  1.5431
#>  1.0000
#>  1.5431
#> [ CPUf32{3} ] 
```
