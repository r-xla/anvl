# Primitive Sign

Element-wise sign.

## Usage

``` r
prim_sign(x)
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

- `reverse`

## StableHLO

Lowers to
[`hlo_sign()`](https://r-xla.github.io/stablehlo/reference/hlo_sign.html),
specified under [sign](https://openxla.org/stablehlo/spec#sign).

## See also

[`nv_sign()`](https://r-xla.github.io/anvl/dev/reference/nv_sign.md),
[`sign()`](https://rdrr.io/r/base/sign.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-3, 0, 5))
prim_sign(x)
#> AnvlArray
#>  -1
#>   0
#>   1
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_sign(-3)
#> AnvlArray
#>  -1
#> [ CPUf32{} ] 
```
