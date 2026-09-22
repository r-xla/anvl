# Primitive Inverse Hyperbolic Sine

Element-wise inverse hyperbolic sine.

## Usage

``` r
prim_asinh(x)
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
[`hlo_asinh()`](https://r-xla.github.io/stablehlo/reference/hlo_asinh.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.asinh](https://openxla.org/stablehlo/generated/chlo#chloasinh_chloasinhop).

## See also

[`nv_asinh()`](https://r-xla.github.io/anvl/dev/reference/nv_asinh.md),
[`asinh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-1, 0, 1))
prim_asinh(x)
#> AnvlArray
#>  -0.8814
#>   0.0000
#>   0.8814
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_asinh(1)
#> AnvlArray
#>  0.8814
#> [ CPUf32{} ] 
```
