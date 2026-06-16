# Primitive Sign

Element-wise sign.

## Usage

``` r
prim_sign(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type signed integer or floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_sign()`](https://r-xla.github.io/stablehlo/reference/hlo_sign.html).

## See also

[`nv_sign()`](https://r-xla.github.io/anvl/dev/reference/nv_sign.md),
[`sign()`](https://rdrr.io/r/base/sign.html)

## Examples

``` r
x <- nv_array(c(-3, 0, 5))
prim_sign(x)
#> AnvlArray
#>  -1
#>   0
#>   1
#> [ CPUf32{3} ] 
```
