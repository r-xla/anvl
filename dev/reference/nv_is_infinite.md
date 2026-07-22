# Is Infinite

Element-wise check if values are infinite (`Inf` or `-Inf`). You can
also use [`is.infinite()`](https://rdrr.io/r/base/is.finite.html).

## Usage

``` r
nv_is_infinite(x)

# S3 method for class 'AnvlArray'
is.infinite(x)
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
[`nv_is_nan()`](https://r-xla.github.io/anvl/dev/reference/nv_is_nan.md)

## Examples

``` r
x <- nv_array(c(1, NaN, Inf, -Inf, 0))
nv_is_infinite(x)
#> AnvlArray
#>  0
#>  0
#>  1
#>  1
#>  0
#> [ CPUbool{5} ] 
```
