# Primitive Exponential Minus One

Element-wise exp(x) - 1, more accurate for small x.

## Usage

``` r
prim_expm1(x)
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

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_exponential_minus_one()`](https://r-xla.github.io/stablehlo/reference/hlo_exponential_minus_one.html).

## See also

[`nv_expm1()`](https://r-xla.github.io/anvl/reference/nv_expm1.md)

## Examples

``` r
x <- nv_array(c(0, 0.001, 1))
prim_expm1(x)
#> AnvlArray
#>  0.0000
#>  0.0010
#>  1.7183
#> [ CPUf32{3} ] 
```
