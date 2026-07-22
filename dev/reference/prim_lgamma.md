# Primitive Log-Gamma

Element-wise natural logarithm of the absolute value of the gamma
function.

## Usage

``` r
prim_lgamma(x)
```

## Arguments

- x:

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
[`hlo_lgamma()`](https://r-xla.github.io/stablehlo/reference/hlo_lgamma.html).

## See also

[`nv_lgamma()`](https://r-xla.github.io/anvl/dev/reference/nv_lgamma.md),
[`lgamma()`](https://rdrr.io/r/base/Special.html)

## Examples

``` r
x <- nv_array(c(0.5, 1, 2, 5))
prim_lgamma(x)
#> AnvlArray
#>  5.7236e-01
#>  4.7684e-07
#>  0.0000e+00
#>  3.1781e+00
#> [ CPUf32{4} ] 
```
