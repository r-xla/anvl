# Primitive Hyperbolic Sine

Element-wise hyperbolic sine.

## Usage

``` r
prim_sinh(x)
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
[`hlo_sinh()`](https://r-xla.github.io/stablehlo/reference/hlo_sinh.html).

## See also

[`nv_sinh()`](https://r-xla.github.io/anvl/dev/reference/nv_sinh.md),
[`sinh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_sinh(x)
#> AnvlArray
#>  -1.1752
#>   0.0000
#>   1.1752
#> [ CPUf32{3} ] 
```
