# Inverse Hyperbolic Sine

Element-wise inverse hyperbolic sine. You can also use
[`asinh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nv_asinh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_asinh()`](https://r-xla.github.io/anvl/dev/reference/prim_asinh.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
asinh(x)
#> AnvlArray
#>  -0.8814
#>   0.0000
#>   0.8814
#> [ CPUf32{3} ] 
```
