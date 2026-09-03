# Primitive Ceiling

Element-wise ceiling.

## Usage

``` r
prim_ceil(x)
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

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_ceil()`](https://r-xla.github.io/stablehlo/reference/hlo_ceil.html).

## See also

[`nv_ceiling()`](https://r-xla.github.io/anvl/dev/reference/nv_ceiling.md),
[`ceiling()`](https://rdrr.io/r/base/Round.html)

## Examples

``` r
x <- nv_array(c(1.2, 2.7, -1.5))
prim_ceil(x)
#> AnvlArray
#>   2
#>   3
#>  -1
#> [ CPUf32{3} ] 
```
