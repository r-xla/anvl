# Primitive Cube Root

Element-wise cube root.

## Usage

``` r
prim_cbrt(x)
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

- `reverse`

## StableHLO

Lowers to
[`hlo_cbrt()`](https://r-xla.github.io/stablehlo/reference/hlo_cbrt.html),
specified under [cbrt](https://openxla.org/stablehlo/spec#cbrt).

## See also

[`nv_cbrt()`](https://r-xla.github.io/anvl/dev/reference/nv_cbrt.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, 8, 27))
prim_cbrt(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_cbrt(8)
#> AnvlArray
#>  2
#> [ CPUf32{} ] 
```
