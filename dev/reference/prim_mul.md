# Primitive Multiplication

Multiplies two arrays element-wise.

## Usage

``` r
prim_mul(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_multiply()`](https://r-xla.github.io/stablehlo/reference/hlo_multiply.html).

## See also

[`nv_mul()`](https://r-xla.github.io/anvl/dev/reference/nv_mul.md), `*`

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
prim_mul(x, y)
#> AnvlArray
#>   4
#>  10
#>  18
#> [ CPUf32{3} ] 
```
