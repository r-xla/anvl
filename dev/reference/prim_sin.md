# Primitive Sine

Element-wise sine.

## Usage

``` r
prim_sin(x)
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
[`hlo_sine()`](https://r-xla.github.io/stablehlo/reference/hlo_sine.html),
specified under [sine](https://openxla.org/stablehlo/spec#sine).

## See also

[`nv_sin()`](https://r-xla.github.io/anvl/dev/reference/nv_sin.md),
[`sin()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, pi / 2, pi))
prim_sin(x)
#> AnvlArray
#>   0.0000e+00
#>   1.0000e+00
#>  -8.7423e-08
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_sin(0)
#> AnvlArray
#>  0
#> [ CPUf32{} ] 
```
