# Primitive Square Root

Element-wise square root.

## Usage

``` r
prim_sqrt(x)
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

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_sqrt()`](https://r-xla.github.io/stablehlo/reference/hlo_sqrt.html).

## See also

[`nv_sqrt()`](https://r-xla.github.io/anvl/dev/reference/nv_sqrt.md),
[`sqrt()`](https://rdrr.io/r/base/MathFun.html)

## Examples

``` r
x <- nv_array(c(1, 4, 9))
prim_sqrt(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
