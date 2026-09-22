# Primitive Cosine

Element-wise cosine.

## Usage

``` r
prim_cos(x)
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
[`hlo_cosine()`](https://r-xla.github.io/stablehlo/reference/hlo_cosine.html),
specified under [cosine](https://openxla.org/stablehlo/spec#cosine).

## See also

[`nv_cos()`](https://r-xla.github.io/anvl/dev/reference/nv_cos.md),
[`cos()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, pi / 2, pi))
prim_cos(x)
#> AnvlArray
#>   1.0000e+00
#>  -4.3711e-08
#>  -1.0000e+00
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_cos(0)
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
