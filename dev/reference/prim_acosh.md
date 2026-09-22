# Primitive Inverse Hyperbolic Cosine

Element-wise inverse hyperbolic cosine.

## Usage

``` r
prim_acosh(x)
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
[`hlo_acosh()`](https://r-xla.github.io/stablehlo/reference/hlo_acosh.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.acosh](https://openxla.org/stablehlo/generated/chlo#chloacosh_chloacoshop).

## See also

[`nv_acosh()`](https://r-xla.github.io/anvl/dev/reference/nv_acosh.md),
[`acosh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, 2, 10))
prim_acosh(x)
#> AnvlArray
#>  0.0000
#>  1.3170
#>  2.9932
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_acosh(2)
#> AnvlArray
#>  1.3170
#> [ CPUf32{} ] 
```
