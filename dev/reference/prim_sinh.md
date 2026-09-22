# Primitive Hyperbolic Sine

Element-wise hyperbolic sine.

## Usage

``` r
prim_sinh(x)
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
[`hlo_sinh()`](https://r-xla.github.io/stablehlo/reference/hlo_sinh.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.sinh](https://openxla.org/stablehlo/generated/chlo#chlosinh_chlosinhop).

## See also

[`nv_sinh()`](https://r-xla.github.io/anvl/dev/reference/nv_sinh.md),
[`sinh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_sinh(x)
#> AnvlArray
#>  -1.1752
#>   0.0000
#>   1.1752
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_sinh(1)
#> AnvlArray
#>  1.1752
#> [ CPUf32{} ] 
```
