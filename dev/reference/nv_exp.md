# Exponential

Element-wise exponential. You can also use
[`exp()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_exp(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_exp()`](https://r-xla.github.io/anvl/dev/reference/prim_exp.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(0, 1, 2))
exp(x)
#> AnvlArray
#>  1.0000
#>  2.7183
#>  7.3891
#> [ CPUf32{3} ] 
```
