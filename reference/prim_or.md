# Primitive Or

Element-wise logical OR.

## Usage

``` r
prim_or(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish values of data type boolean, integer, or unsigned integer.
  Must have the same shape.

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
[`hlo_or()`](https://r-xla.github.io/stablehlo/reference/hlo_or.html).

## See also

[`nv_or()`](https://r-xla.github.io/anvl/reference/nv_or.md), `|`

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
y <- nv_array(c(TRUE, TRUE, FALSE))
prim_or(x, y)
#> AnvlArray
#>  1
#>  1
#>  1
#> [ CPUbool{3} ] 
```
