# Inverse Hyperbolic Tangent

Element-wise inverse hyperbolic tangent. You can also use
[`atanh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nv_atanh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_atanh()`](https://r-xla.github.io/anvl/reference/prim_atanh.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-0.5, 0, 0.5))
atanh(x)
#> AnvlArray
#>  -0.5493
#>   0.0000
#>   0.5493
#> [ CPUf32{3} ] 
```
