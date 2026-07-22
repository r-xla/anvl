# Primitive Absolute Value

Element-wise absolute value.

## Usage

``` r
prim_abs(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type signed integer or floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_abs()`](https://r-xla.github.io/stablehlo/reference/hlo_abs.html).

## See also

[`nv_abs()`](https://r-xla.github.io/anvl/dev/reference/nv_abs.md),
[`abs()`](https://rdrr.io/r/base/MathFun.html)

## Examples

``` r
x <- nv_array(c(-1, 2, -3))
prim_abs(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
