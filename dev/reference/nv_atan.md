# Arc Tangent

Element-wise inverse tangent. You can also use
[`atan()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_atan(x)
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

[`prim_atan()`](https://r-xla.github.io/anvl/dev/reference/prim_atan.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
atan(x)
#> AnvlArray
#>  -0.7854
#>   0.0000
#>   0.7854
#> [ CPUf32{3} ] 
```
