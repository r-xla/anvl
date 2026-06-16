# Primitive Sine

Element-wise sine.

## Usage

``` r
prim_sin(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

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
[`hlo_sine()`](https://r-xla.github.io/stablehlo/reference/hlo_sine.html).

## See also

[`nv_sin()`](https://r-xla.github.io/anvl/dev/reference/nv_sin.md),
[`sin()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
x <- nv_array(c(0, pi / 2, pi))
prim_sin(x)
#> AnvlArray
#>   0.0000e+00
#>   1.0000e+00
#>  -8.7423e-08
#> [ CPUf32{3} ] 
```
