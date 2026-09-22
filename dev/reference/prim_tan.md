# Primitive Tangent

Element-wise tangent.

## Usage

``` r
prim_tan(x)
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
[`hlo_tan()`](https://r-xla.github.io/stablehlo/reference/hlo_tan.html),
specified under [tan](https://openxla.org/stablehlo/spec#tan).

## See also

[`nv_tan()`](https://r-xla.github.io/anvl/dev/reference/nv_tan.md),
[`tan()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, 0.5, 1))
prim_tan(x)
#> AnvlArray
#>  0.0000
#>  0.5463
#>  1.5574
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_tan(0.5)
#> AnvlArray
#>  0.5463
#> [ CPUf32{} ] 
```
