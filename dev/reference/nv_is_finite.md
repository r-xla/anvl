# Is Finite

Element-wise check if values are finite (not `Inf`, `-Inf`, or `NaN`).

## Usage

``` r
nv_is_finite(x)

# S3 method for class 'AnvlArray'
is.finite(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape as the input and boolean data type.

## See also

[`prim_is_finite()`](https://r-xla.github.io/anvl/dev/reference/prim_is_finite.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, Inf, NaN, -Inf, 0))
nv_is_finite(x)
#> AnvlArray
#>  1
#>  0
#>  0
#>  0
#>  1
#> [ CPUbool{5} ] 
```
