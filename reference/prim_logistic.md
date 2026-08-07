# Primitive Logistic (Sigmoid)

Element-wise logistic sigmoid: 1 / (1 + exp(-x)).

## Usage

``` r
prim_logistic(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_logistic()`](https://r-xla.github.io/stablehlo/reference/hlo_logistic.html).

## See also

[`nv_logistic()`](https://r-xla.github.io/anvl/reference/nv_logistic.md)

## Examples

``` r
x <- nv_array(c(-2, 0, 2))
prim_logistic(x)
#> AnvlArray
#>  0.1192
#>  0.5000
#>  0.8808
#> [ CPUf32{3} ] 
```
