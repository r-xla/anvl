# Primitive Greater Than

Element-wise greater than comparison.

## Usage

``` r
prim_gt(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape as the inputs and boolean data type. It is ambiguous
if both inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html)
with `comparison_direction = "GT"`.

## See also

[`nv_gt()`](https://r-xla.github.io/anvl/reference/nv_gt.md), `>`

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(3, 2, 1))
prim_gt(x, y)
#> AnvlArray
#>  0
#>  0
#>  1
#> [ CPUbool{3} ] 
```
