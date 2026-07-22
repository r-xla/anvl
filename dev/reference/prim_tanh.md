# Primitive Hyperbolic Tangent

Element-wise hyperbolic tangent.

## Usage

``` r
prim_tanh(x)
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
[`hlo_tanh()`](https://r-xla.github.io/stablehlo/reference/hlo_tanh.html).

## See also

[`nv_tanh()`](https://r-xla.github.io/anvl/dev/reference/nv_tanh.md),
[`tanh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_tanh(x)
#> AnvlArray
#>  -0.7616
#>   0.0000
#>   0.7616
#> [ CPUf32{3} ] 
```
