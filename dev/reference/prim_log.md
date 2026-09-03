# Primitive Logarithm

Element-wise natural logarithm.

## Usage

``` r
prim_log(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_log()`](https://r-xla.github.io/stablehlo/reference/hlo_log.html).

## See also

[`nv_log()`](https://r-xla.github.io/anvl/dev/reference/nv_log.md),
[`log()`](https://rdrr.io/r/base/Log.html)

## Examples

``` r
x <- nv_array(c(1, 2.718, 7.389))
prim_log(x)
#> AnvlArray
#>  0.0000
#>  0.9999
#>  2.0000
#> [ CPUf32{3} ] 
```
