# Primitive Addition

Adds two arrays element-wise.

## Usage

``` r
prim_add(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_add()`](https://r-xla.github.io/stablehlo/reference/hlo_add.html).

## See also

[`nv_add()`](https://r-xla.github.io/anvl/reference/nv_add.md), `+`

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
prim_add(x, y)
#> AnvlArray
#>  5
#>  7
#>  9
#> [ CPUf32{3} ] 
```
