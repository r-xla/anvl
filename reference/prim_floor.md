# Primitive Floor

Element-wise floor.

## Usage

``` r
prim_floor(x)
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
[`hlo_floor()`](https://r-xla.github.io/stablehlo/reference/hlo_floor.html).

## See also

[`nv_floor()`](https://r-xla.github.io/anvl/reference/nv_floor.md),
[`floor()`](https://rdrr.io/r/base/Round.html)

## Examples

``` r
x <- nv_array(c(1.2, 2.7, -1.5))
prim_floor(x)
#> AnvlArray
#>   1
#>   2
#>  -2
#> [ CPUf32{3} ] 
```
