# Primitive Population Count

Element-wise population count (number of set bits).

## Usage

``` r
prim_popcnt(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type integer or unsigned integer.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_popcnt()`](https://r-xla.github.io/stablehlo/reference/hlo_popcnt.html).

## See also

[`nv_popcnt()`](https://r-xla.github.io/anvl/dev/reference/nv_popcnt.md)

## Examples

``` r
x <- nv_array(c(7L, 3L, 15L))
prim_popcnt(x)
#> AnvlArray
#>  3
#>  2
#>  4
#> [ CPUi32{3} ] 
```
