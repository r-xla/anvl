# Primitive Ceiling

Element-wise ceiling.

## Usage

``` r
prim_ceil(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any float data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_ceil()`](https://r-xla.github.io/stablehlo/reference/hlo_ceil.html),
specified under [ceil](https://openxla.org/stablehlo/spec#ceil).

## See also

[`nv_ceiling()`](https://r-xla.github.io/anvl/dev/reference/nv_ceiling.md),
[`ceiling()`](https://rdrr.io/r/base/Round.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1.2, 2.7, -1.5))
prim_ceil(x)
#> AnvlArray
#>   2
#>   3
#>  -1
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_ceil(1.2)
#> AnvlArray
#>  2
#> [ CPUf32{} ] 
```
