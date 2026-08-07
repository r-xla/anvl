# Convert Data Type

Converts the elements of an array to a different data type. Returns the
input unchanged if it already has the target type.

## Usage

``` r
nv_convert(x, dtype)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the given `dtype` and the same shape as `x`.

## See also

[`prim_convert()`](https://r-xla.github.io/anvl/reference/prim_convert.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1L, 2L, 3L))
nv_convert(x, dtype = "f32")
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
