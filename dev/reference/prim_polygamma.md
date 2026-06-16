# Primitive Polygamma

Element-wise polygamma function: the `(n+1)`-th derivative of the
log-gamma function. Both `n` and `x` must have the same shape; `n`
typically holds non-negative integer values.

## Usage

``` r
prim_polygamma(n, x)
```

## Arguments

- n, x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of data type floating-point. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_polygamma()`](https://r-xla.github.io/stablehlo/reference/hlo_polygamma.html).

## See also

[`nv_polygamma()`](https://r-xla.github.io/anvl/dev/reference/nv_polygamma.md)

## Examples

``` r
n <- nv_array(c(1, 1, 2))
x <- nv_array(c(0.5, 1, 2))
prim_polygamma(n, x)
#> AnvlArray
#>   4.9348
#>   1.6449
#>  -0.4041
#> [ CPUf32{3} ] 
```
