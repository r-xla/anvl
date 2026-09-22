# Primitive Floor

Element-wise floor.

## Usage

``` r
prim_floor(x)
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
[`hlo_floor()`](https://r-xla.github.io/stablehlo/reference/hlo_floor.html),
specified under [floor](https://openxla.org/stablehlo/spec#floor).

## See also

[`nv_floor()`](https://r-xla.github.io/anvl/dev/reference/nv_floor.md),
[`floor()`](https://rdrr.io/r/base/Round.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1.2, 2.7, -1.5))
prim_floor(x)
#> AnvlArray
#>   1
#>   2
#>  -2
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_floor(1.2)
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
