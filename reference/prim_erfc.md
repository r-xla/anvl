# Primitive Complementary Error Function

Element-wise complementary error function `erfc(x) = 1 - erf(x)`.

## Usage

``` r
prim_erfc(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_erfc()`](https://r-xla.github.io/stablehlo/reference/hlo_erfc.html).

## See also

[`nv_erfc()`](https://r-xla.github.io/anvl/reference/nv_erfc.md)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_erfc(x)
#> AnvlArray
#>  1.8427
#>  1.0000
#>  0.1573
#> [ CPUf32{3} ] 
```
