# Is Infinite

Element-wise check if values are infinite (`Inf` or `-Inf`). You can
also use [`is.infinite()`](https://rdrr.io/r/base/is.finite.html).

## Usage

``` r
nv_is_infinite(operand)

# S3 method for class 'AnvlArray'
is.infinite(x)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Same as `operand`; this is the name used by the base R S3 generic.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape as the input and boolean data type.

## See also

[`nv_is_finite()`](https://r-xla.github.io/anvl/reference/nv_is_finite.md),
[`nv_is_nan()`](https://r-xla.github.io/anvl/reference/nv_is_nan.md)

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
