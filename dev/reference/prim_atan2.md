# Primitive Atan2

Element-wise atan2 operation.

## Usage

``` r
prim_atan2(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of data type floating-point. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_atan2()`](https://r-xla.github.io/stablehlo/reference/hlo_atan2.html).

## See also

[`nv_atan2()`](https://r-xla.github.io/anvl/dev/reference/nv_atan2.md)

## Examples

``` r
y <- nv_array(c(1, 0, -1))
x <- nv_array(c(0, 1, 0))
prim_atan2(y, x)
#> AnvlArray
#>   1.5708
#>   0.0000
#>  -1.5708
#> [ CPUf32{3} ] 
```
