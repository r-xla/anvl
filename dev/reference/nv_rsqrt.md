# Reciprocal Square Root

Element-wise reciprocal square root, i.e. `1 / sqrt(x)`.

## Usage

``` r
nv_rsqrt(x)
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

[`prim_rsqrt()`](https://r-xla.github.io/anvl/dev/reference/prim_rsqrt.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 4, 9))
nv_rsqrt(x)
#> AnvlArray
#>  1.0000
#>  0.5000
#>  0.3333
#> [ CPUf32{3} ] 
```
