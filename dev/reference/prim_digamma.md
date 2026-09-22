# Primitive Digamma

Element-wise digamma function (logarithmic derivative of the gamma
function).

## Usage

``` r
prim_digamma(x)
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
[`hlo_digamma()`](https://r-xla.github.io/stablehlo/reference/hlo_digamma.html),
an op of the CHLO dialect, a higher-level companion to StableHLO that is
lowered to it during compilation. See
[chlo.digamma](https://openxla.org/stablehlo/generated/chlo#chlodigamma_chlodigammaop).

## See also

[`nv_digamma()`](https://r-xla.github.io/anvl/dev/reference/nv_digamma.md),
[`digamma()`](https://rdrr.io/r/base/Special.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0.5, 1, 2, 5))
prim_digamma(x)
#> AnvlArray
#>  -1.9635
#>  -0.5772
#>   0.4228
#>   1.5061
#> [ CPUf32{4} ] 

# an R value materializes at its default data type
prim_digamma(2)
#> AnvlArray
#>  0.4228
#> [ CPUf32{} ] 
```
