# Log-Gamma

Element-wise natural logarithm of the absolute value of the gamma
function. You can also use
[`lgamma()`](https://rdrr.io/r/base/Special.html).

## Usage

``` r
nv_lgamma(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_lgamma()`](https://r-xla.github.io/anvl/reference/prim_lgamma.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(0.5, 1, 2, 5))
lgamma(x)
#> AnvlArray
#>  5.7236e-01
#>  4.7684e-07
#>  0.0000e+00
#>  3.1781e+00
#> [ CPUf32{4} ] 
```
