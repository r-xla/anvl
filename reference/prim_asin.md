# Primitive Arc Sine

Element-wise inverse sine.

## Usage

``` r
prim_asin(x)
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
[`hlo_asin()`](https://r-xla.github.io/stablehlo/reference/hlo_asin.html).

## See also

[`nv_asin()`](https://r-xla.github.io/anvl/reference/nv_asin.md),
[`asin()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_asin(x)
#> AnvlArray
#>  -1.5708
#>   0.0000
#>   1.5708
#> [ CPUf32{3} ] 
```
