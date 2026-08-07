# Hyperbolic Sine

Element-wise hyperbolic sine. You can also use
[`sinh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nv_sinh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_sinh()`](https://r-xla.github.io/anvl/reference/prim_sinh.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
sinh(x)
#> AnvlArray
#>  -1.1752
#>   0.0000
#>   1.1752
#> [ CPUf32{3} ] 
```
