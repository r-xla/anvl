# Arc Cosine

Element-wise inverse cosine. You can also use
[`acos()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_acos(x)
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

[`prim_acos()`](https://r-xla.github.io/anvl/dev/reference/prim_acos.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0, 1))
acos(x)
#> AnvlArray
#>  3.1416
#>  1.5708
#>  0.0000
#> [ CPUf32{3} ] 
```
