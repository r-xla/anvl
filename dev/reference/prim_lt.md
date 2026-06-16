# Primitive Less Than

Element-wise less than comparison.

## Usage

``` r
prim_lt(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape as the inputs and boolean data type. It is ambiguous
if both inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html)
with `comparison_direction = "LT"`.

## See also

[`nv_lt()`](https://r-xla.github.io/anvl/dev/reference/nv_lt.md), `<`

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(3, 2, 1))
prim_lt(x, y)
#> AnvlArray
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 
```
