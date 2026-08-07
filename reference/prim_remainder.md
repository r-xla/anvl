# Primitive Remainder

Element-wise remainder. Result has sign of the divident, which differs
from base R's `%%`, which is available via
[`nv_mod()`](https://r-xla.github.io/anvl/reference/nv_mod.md) and has
sign of divisor.

## Usage

``` r
prim_remainder(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish values of data type integer, unsigned integer, or
  floating-point. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_remainder()`](https://r-xla.github.io/stablehlo/reference/hlo_remainder.html).

## See also

[`nv_remainder()`](https://r-xla.github.io/anvl/reference/nv_remainder.md)

## Examples

``` r
prim_remainder(1, -3)
#> AnvlArray
#>  1
#> [ CPUf32?{} ] 
1 %% -3
#> [1] -2
```
