# Primitive Error Function

Element-wise error function
`erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt`.

## Usage

``` r
prim_erf(operand)
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

- `reverse`

## StableHLO

Lowers to
[`hlo_erf()`](https://r-xla.github.io/stablehlo/reference/hlo_erf.html).

## See also

[`nv_erf()`](https://r-xla.github.io/anvl/dev/reference/nv_erf.md)

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
prim_erf(x)
#> AnvlArray
#>  -0.8427
#>   0.0000
#>   0.8427
#> [ CPUf32{3} ] 
```
