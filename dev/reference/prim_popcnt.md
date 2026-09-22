# Primitive Population Count

Element-wise population count (number of set bits).

## Usage

``` r
prim_popcnt(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any integer data type. An R value materializes at
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_popcnt()`](https://r-xla.github.io/stablehlo/reference/hlo_popcnt.html),
specified under [popcnt](https://openxla.org/stablehlo/spec#popcnt).

## See also

[`nv_popcnt()`](https://r-xla.github.io/anvl/dev/reference/nv_popcnt.md)

## Examples

``` r
# the set bits are counted, at the input's own integer data type
x <- nv_array(c(7L, 3L, 15L))
prim_popcnt(x)
#> AnvlArray
#>  3
#>  2
#>  4
#> [ CPUi32{3} ] 
```
