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

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array. An integer array is converted to the default floating
  point type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape as the input, and its data type – or the default
float data type if the input was an integer array.

## See also

[`prim_lgamma()`](https://r-xla.github.io/anvl/dev/reference/prim_lgamma.md)
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
