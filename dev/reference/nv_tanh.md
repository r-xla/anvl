# Hyperbolic Tangent

Element-wise hyperbolic tangent. You can also use
[`tanh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nv_tanh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_tanh()`](https://r-xla.github.io/anvl/dev/reference/prim_tanh.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
tanh(x)
#> AnvlArray
#>  -0.7616
#>   0.0000
#>   0.7616
#> [ CPUf32{3} ] 
```
