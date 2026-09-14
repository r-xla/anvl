# Cube Root

Element-wise cube root.

## Usage

``` r
nv_cbrt(x)
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

[`prim_cbrt()`](https://r-xla.github.io/anvl/dev/reference/prim_cbrt.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 8, 27))
nv_cbrt(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
