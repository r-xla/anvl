# Is NaN

Element-wise check if values are NaN. You can also use
[`is.nan()`](https://rdrr.io/r/base/is.finite.html). Only a float holds
a NaN, so the answer for any other data type is all `FALSE` and is built
as a constant rather than computed.

## Usage

``` r
nv_is_nan(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and boolean data type.

## See also

[`nv_is_finite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_finite.md),
[`nv_is_infinite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_infinite.md)

## Examples

``` r
# a boolean result, whatever the input's data type
x <- nv_array(c(1, NaN, Inf, -Inf, 0))
nv_is_nan(x)
#> AnvlArray
#>  0
#>  1
#>  0
#>  0
#>  0
#> [ CPUbool{5} ] 

# all FALSE for an integer input, which has no NaN to find
nv_is_nan(nv_array(1:3))
#> AnvlArray
#>  0
#>  0
#>  0
#> [ CPUbool{3} ] 
```
