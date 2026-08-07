# Tangent

Element-wise tangent. You can also use
[`tan()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_tan(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_tan()`](https://r-xla.github.io/anvl/reference/prim_tan.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(0, 0.5, 1))
tan(x)
#> AnvlArray
#>  0.0000
#>  0.5463
#>  1.5574
#> [ CPUf32{3} ] 
```
