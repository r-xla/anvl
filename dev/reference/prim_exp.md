# Primitive Exponential

Element-wise exponential.

## Usage

``` r
prim_exp(x)
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

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_exponential()`](https://r-xla.github.io/stablehlo/reference/hlo_exponential.html).

## See also

[`nv_exp()`](https://r-xla.github.io/anvl/dev/reference/nv_exp.md),
[`exp()`](https://rdrr.io/r/base/Log.html)

## Examples

``` r
x <- nv_array(c(0, 1, 2))
prim_exp(x)
#> AnvlArray
#>  1.0000
#>  2.7183
#>  7.3891
#> [ CPUf32{3} ] 
```
