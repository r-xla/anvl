# Primitive Digamma

Element-wise digamma function (logarithmic derivative of the gamma
function).

## Usage

``` r
prim_digamma(operand)
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
[`stablehlo::hlo_digamma()`](https://r-xla.github.io/stablehlo/reference/hlo_digamma.html).

## See also

[`nv_digamma()`](https://r-xla.github.io/anvl/reference/nv_digamma.md),
[`digamma()`](https://rdrr.io/r/base/Special.html)

## Examples

``` r
x <- nv_array(c(0.5, 1, 2, 5))
prim_digamma(x)
#> AnvlArray
#>  -1.9635
#>  -0.5772
#>   0.4228
#>   1.5061
#> [ CPUf32{4} ] 
```
