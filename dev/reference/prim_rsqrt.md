# Primitive Reciprocal Square Root

Element-wise reciprocal square root.

## Usage

``` r
prim_rsqrt(operand)
```

## Arguments

- operand:

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
[`hlo_rsqrt()`](https://r-xla.github.io/stablehlo/reference/hlo_rsqrt.html).

## See also

[`nv_rsqrt()`](https://r-xla.github.io/anvl/dev/reference/nv_rsqrt.md)

## Examples

``` r
x <- nv_array(c(1, 4, 9))
prim_rsqrt(x)
#> AnvlArray
#>  1.0000
#>  0.5000
#>  0.3333
#> [ CPUf32{3} ] 
```
