# Cosine

Element-wise cosine. You can also use
[`cos()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_cos(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_cos()`](https://r-xla.github.io/anvl/reference/prim_cos.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(0, pi / 2, pi))
cos(x)
#> AnvlArray
#>   1.0000e+00
#>  -4.3711e-08
#>  -1.0000e+00
#> [ CPUf32{3} ] 
```
