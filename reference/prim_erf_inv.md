# Primitive Inverse Error Function

Element-wise inverse error function (the inverse of `erf` on `(-1, 1)`).

## Usage

``` r
prim_erf_inv(operand)
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
[`stablehlo::hlo_erf_inv()`](https://r-xla.github.io/stablehlo/reference/hlo_erf_inv.html).

## See also

[`nv_erf_inv()`](https://r-xla.github.io/anvl/reference/nv_erf_inv.md)

## Examples

``` r
x <- nv_array(c(-0.5, 0, 0.5))
prim_erf_inv(x)
#> AnvlArray
#>  -0.4769
#>   0.0000
#>   0.4769
#> [ CPUf32{3} ] 
```
