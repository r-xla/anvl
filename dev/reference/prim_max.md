# Primitive Maximum

Element-wise maximum of two arrays.

## Usage

``` r
prim_max(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_maximum()`](https://r-xla.github.io/stablehlo/reference/hlo_maximum.html).

## See also

[`nv_max()`](https://r-xla.github.io/anvl/dev/reference/nv_max.md)

## Examples

``` r
x <- nv_array(c(1, 5, 3))
y <- nv_array(c(4, 2, 6))
prim_max(x, y)
#> AnvlArray
#>  4
#>  5
#>  6
#> [ CPUf32{3} ] 
```
