# Tangent of a Multiple of Pi

Element-wise `tan(pi * x)`. You can also use
[`tanpi()`](https://rdrr.io/r/base/Trig.html). Like base R's
[`base::tanpi()`](https://rdrr.io/r/base/Trig.html), it is exact for a
whole argument and `NaN` at the half integers, where the tangent has its
poles.

## Usage

``` r
nv_tanpi(x)
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

[`nv_sinpi()`](https://r-xla.github.io/anvl/dev/reference/nv_sinpi.md),
[`nv_cospi()`](https://r-xla.github.io/anvl/dev/reference/nv_cospi.md),
[`nv_tan()`](https://r-xla.github.io/anvl/dev/reference/nv_tan.md)

## Examples

``` r
tanpi(nv_array(c(0, 0.25, 0.5, 1)))
#> AnvlArray
#>    0
#>    1
#>  nan
#>    0
#> [ CPUf32{4} ] 
```
