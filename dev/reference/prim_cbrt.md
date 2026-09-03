# Primitive Cube Root

Element-wise cube root.

## Usage

``` r
prim_cbrt(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_cbrt()`](https://r-xla.github.io/stablehlo/reference/hlo_cbrt.html).

## See also

[`nv_cbrt()`](https://r-xla.github.io/anvl/dev/reference/nv_cbrt.md)

## Examples

``` r
x <- nv_array(c(1, 8, 27))
prim_cbrt(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
