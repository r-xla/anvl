# Primitive Logistic (Sigmoid)

Element-wise logistic sigmoid: 1 / (1 + exp(-x)).

## Usage

``` r
prim_logistic(x)
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

[`nv_logistic()`](https://r-xla.github.io/anvl/dev/reference/nv_logistic.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-2, 0, 2))
prim_logistic(x)
#> AnvlArray
#>  0.1192
#>  0.5000
#>  0.8808
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_logistic(2)
#> AnvlArray
#>  0.8808
#> [ CPUf32{} ] 
```
