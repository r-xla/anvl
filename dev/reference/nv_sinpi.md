# Sine of a Multiple of Pi

Element-wise `sin(pi * x)`. You can also use
[`sinpi()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_sinpi(x)
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

[`nv_cospi()`](https://r-xla.github.io/anvl/dev/reference/nv_cospi.md),
[`nv_tanpi()`](https://r-xla.github.io/anvl/dev/reference/nv_tanpi.md),
[`nv_sin()`](https://r-xla.github.io/anvl/dev/reference/nv_sin.md)

## Examples

``` r
sinpi(nv_array(c(0, 0.5, 1, 1.5)))
#> AnvlArray
#>   0
#>   1
#>   0
#>  -1
#> [ CPUf32{4} ] 
```
