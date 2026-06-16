# Primitive Subtraction

Subtracts two arrays element-wise.

## Usage

``` r
prim_sub(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of data type integer, unsigned integer, or
  floating-point. Must have the same shape.

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
[`hlo_subtract()`](https://r-xla.github.io/stablehlo/reference/hlo_subtract.html).

## See also

[`nv_sub()`](https://r-xla.github.io/anvl/dev/reference/nv_sub.md), `-`

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
prim_sub(x, y)
#> AnvlArray
#>  -3
#>  -3
#>  -3
#> [ CPUf32{3} ] 
```
