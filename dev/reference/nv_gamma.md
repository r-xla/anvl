# Gamma Function

Element-wise gamma function. You can also use
[`gamma()`](https://rdrr.io/r/base/Special.html).

## Usage

``` r
nv_gamma(x)
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

[`nv_lgamma()`](https://r-xla.github.io/anvl/dev/reference/nv_lgamma.md),
which is what the hardware computes.

## Examples

``` r
gamma(nv_array(c(0.5, 1, 5, -1.5)))
#> AnvlArray
#>   1.7725
#>   1.0000
#>  24.0000
#>   2.3633
#> [ CPUf32{4} ] 
```
