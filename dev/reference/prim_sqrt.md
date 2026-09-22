# Primitive Square Root

Element-wise square root.

## Usage

``` r
prim_sqrt(x)
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
[`hlo_sqrt()`](https://r-xla.github.io/stablehlo/reference/hlo_sqrt.html),
specified under [sqrt](https://openxla.org/stablehlo/spec#sqrt).

## See also

[`nv_sqrt()`](https://r-xla.github.io/anvl/dev/reference/nv_sqrt.md),
[`sqrt()`](https://rdrr.io/r/base/MathFun.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, 4, 9))
prim_sqrt(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_sqrt(4)
#> AnvlArray
#>  2
#> [ CPUf32{} ] 
```
