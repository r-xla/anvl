# Primitive Logistic (Sigmoid)

Element-wise logistic sigmoid: 1 / (1 + exp(-x)), i.e.
[`stats::plogis()`](https://rdrr.io/r/stats/Logistic.html) with the
default location and scale.

## Usage

``` r
prim_plogis(x)
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
[`hlo_logistic()`](https://r-xla.github.io/stablehlo/reference/hlo_logistic.html),
specified under [logistic](https://openxla.org/stablehlo/spec#logistic).

## See also

[`nv_plogis()`](https://r-xla.github.io/anvl/dev/reference/nv_plogis.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-2, 0, 2))
prim_plogis(x)
#> AnvlArray
#>  0.1192
#>  0.5000
#>  0.8808
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_plogis(2)
#> AnvlArray
#>  0.8808
#> [ CPUf32{} ] 
```
