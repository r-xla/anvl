# Primitive Cosine

Element-wise cosine.

## Usage

``` r
prim_cos(operand)
```

## Arguments

- operand:

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
[`stablehlo::hlo_cosine()`](https://r-xla.github.io/stablehlo/reference/hlo_cosine.html).

## See also

[`nv_cos()`](https://r-xla.github.io/anvl/reference/nv_cos.md),
[`cos()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
x <- nv_array(c(0, pi / 2, pi))
prim_cos(x)
#> AnvlArray
#>   1.0000e+00
#>  -4.3711e-08
#>  -1.0000e+00
#> [ CPUf32{3} ] 
```
