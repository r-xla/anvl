# Tangent

Element-wise tangent. You can also use
[`tan()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_tan(x)
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

[`prim_tan()`](https://r-xla.github.io/anvl/dev/reference/prim_tan.md)
for the underlying primitive.

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
