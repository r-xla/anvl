# Hyperbolic Cosine

Element-wise hyperbolic cosine. You can also use
[`cosh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nv_cosh(x)
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

[`prim_cosh()`](https://r-xla.github.io/anvl/dev/reference/prim_cosh.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
cosh(x)
#> AnvlArray
#>  1.5431
#>  1.0000
#>  1.5431
#> [ CPUf32{3} ] 
```
