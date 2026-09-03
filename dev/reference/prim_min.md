# Primitive Minimum

Element-wise minimum of two arrays.

## Usage

``` r
prim_min(lhs, rhs)
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
[`hlo_minimum()`](https://r-xla.github.io/stablehlo/reference/hlo_minimum.html).

## See also

[`nv_min()`](https://r-xla.github.io/anvl/dev/reference/nv_min.md)

## Examples

``` r
x <- nv_array(c(1, 5, 3))
y <- nv_array(c(4, 2, 6))
prim_min(x, y)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
