# Primitive Negation

Negates an array element-wise.

## Usage

``` r
prim_negate(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any numeric data type. An R value materializes at
  its [default data
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
[`hlo_negate()`](https://r-xla.github.io/stablehlo/reference/hlo_negate.html),
specified under [negate](https://openxla.org/stablehlo/spec#negate).

## See also

[`nv_negate()`](https://r-xla.github.io/anvl/dev/reference/nv_negate.md),
unary `-`

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, -2, 3))
prim_negate(x)
#> AnvlArray
#>  -1
#>   2
#>  -3
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_negate(1)
#> AnvlArray
#>  -1
#> [ CPUf32{} ] 
```
