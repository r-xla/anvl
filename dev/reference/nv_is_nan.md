# Is NaN

Element-wise check if values are NaN. You can also use
[`is.nan()`](https://rdrr.io/r/base/is.finite.html).

## Usage

``` r
nv_is_nan(x)

# S3 method for class 'AnvlArray'
is.nan(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape as the input and boolean data type.

## See also

[`nv_is_finite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_finite.md),
[`nv_is_infinite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_infinite.md)

## Examples

``` r
x <- nv_array(c(1, NaN, Inf, -Inf, 0))
nv_is_nan(x)
#> AnvlArray
#>  0
#>  1
#>  0
#>  0
#>  0
#> [ CPUbool{5} ] 
```
