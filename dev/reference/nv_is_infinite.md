# Is Infinite

Element-wise check if values are infinite (`Inf` or `-Inf`). You can
also use [`is.infinite()`](https://rdrr.io/r/base/is.finite.html). Only
a float holds an infinity, so the answer for any other data type is all
`FALSE` and is built as a constant rather than computed.

## Usage

``` r
nv_is_infinite(x)
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
[`nv_is_nan()`](https://r-xla.github.io/anvl/dev/reference/nv_is_nan.md)

## Examples

``` r
# the result is boolean, whatever float data type the input has
x <- nv_array(c(1, NaN, Inf, -Inf, 0))
nv_is_infinite(x)
#> AnvlArray
#>  0
#>  0
#>  1
#>  1
#>  0
#> [ CPUbool{5} ] 

# all FALSE for an integer input, which has no infinity
nv_is_infinite(nv_array(1:3))
#> AnvlArray
#>  0
#>  0
#>  0
#> [ CPUbool{3} ] 

# an R value materializes at its default data type before the test
nv_is_infinite(1)
#> AnvlArray
#>  0
#> [ CPUbool{} ] 
```
