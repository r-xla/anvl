# Primitive Negation

Negates an array element-wise.

## Usage

``` r
prim_negate(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type integer or floating-point.

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
[`hlo_negate()`](https://r-xla.github.io/stablehlo/reference/hlo_negate.html).

## See also

[`nv_negate()`](https://r-xla.github.io/anvl/dev/reference/nv_negate.md),
unary `-`

## Examples

``` r
x <- nv_array(c(1, -2, 3))
prim_negate(x)
#> AnvlArray
#>  -1
#>   2
#>  -3
#> [ CPUf32{3} ] 
```
