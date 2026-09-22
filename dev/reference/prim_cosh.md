# Primitive Hyperbolic Cosine

Element-wise hyperbolic cosine.

## Usage

``` r
prim_cosh(x)
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
[`hlo_cosh()`](https://r-xla.github.io/stablehlo/reference/hlo_cosh.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.cosh](https://openxla.org/stablehlo/generated/chlo#chlocosh_chlocoshop).

## See also

[`nv_cosh()`](https://r-xla.github.io/anvl/dev/reference/nv_cosh.md),
[`cosh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_cosh(x)
#> AnvlArray
#>  1.5431
#>  1.0000
#>  1.5431
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_cosh(1)
#> AnvlArray
#>  1.5431
#> [ CPUf32{} ] 
```
