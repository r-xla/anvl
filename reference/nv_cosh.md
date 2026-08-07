# Hyperbolic Cosine

Element-wise hyperbolic cosine. You can also use
[`cosh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nv_cosh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_cosh()`](https://r-xla.github.io/anvl/reference/prim_cosh.md) for
the underlying primitive.

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
