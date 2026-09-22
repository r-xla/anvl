# Primitive Absolute Value

Element-wise absolute value.

## Usage

``` r
prim_abs(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any signed numeric data type. An R value
  materializes at its [default data
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
[`hlo_abs()`](https://r-xla.github.io/stablehlo/reference/hlo_abs.html),
specified under [abs](https://openxla.org/stablehlo/spec#abs).

## See also

[`nv_abs()`](https://r-xla.github.io/anvl/dev/reference/nv_abs.md),
[`abs()`](https://rdrr.io/r/base/MathFun.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 2, -3))
prim_abs(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_abs(-1)
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
