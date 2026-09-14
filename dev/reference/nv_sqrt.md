# Square Root

Element-wise square root. You can also use
[`sqrt()`](https://rdrr.io/r/base/MathFun.html).

## Usage

``` r
nv_sqrt(x)
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

[`prim_sqrt()`](https://r-xla.github.io/anvl/dev/reference/prim_sqrt.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 4, 9))
sqrt(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
