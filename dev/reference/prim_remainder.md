# Primitive Remainder

Element-wise remainder. The result has the sign of the dividend, which
is what StableHLO's `remainder` does. Base R's `%%` takes the sign of
the divisor instead and is available via
[`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md).

## Usage

``` r
prim_remainder(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of data type integer, unsigned integer, or
  floating-point. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_remainder()`](https://r-xla.github.io/stablehlo/reference/hlo_remainder.html).

## See also

[`nv_remainder()`](https://r-xla.github.io/anvl/dev/reference/nv_remainder.md)

## Examples

``` r
prim_remainder(1, -3)
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
1 %% -3
#> [1] -2
```
