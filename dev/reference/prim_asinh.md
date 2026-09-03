# Primitive Inverse Hyperbolic Sine

Element-wise inverse hyperbolic sine.

## Usage

``` r
prim_asinh(x)
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

- `reverse`

## StableHLO

Lowers to
[`hlo_asinh()`](https://r-xla.github.io/stablehlo/reference/hlo_asinh.html).

## See also

[`nv_asinh()`](https://r-xla.github.io/anvl/dev/reference/nv_asinh.md),
[`asinh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_asinh(x)
#> AnvlArray
#>  -0.8814
#>   0.0000
#>   0.8814
#> [ CPUf32{3} ] 
```
