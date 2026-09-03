# Primitive Greater Equal

Element-wise greater than or equal comparison.

## Usage

``` r
prim_ge(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape as the inputs and boolean data type.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html)
with `comparison_direction = "GE"`.

## See also

[`nv_ge()`](https://r-xla.github.io/anvl/dev/reference/nv_ge.md), `>=`

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(3, 2, 1))
prim_ge(x, y)
#> AnvlArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 
```
