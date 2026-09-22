# Convert Data Type

Converts the elements of an array to a different data type. Note that R
objects are handled differently than `AnvlArray` inputs. For R objects,
we check whether the requested data type can meaningfully hold the
provided data and otherwise err. For `AnvlArray` inputs such a check is
*not* performed.

## Usage

``` r
nv_convert(x, dtype)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The input to convert.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Target data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the given `dtype` and the input's shape.

## See also

[`prim_convert()`](https://r-xla.github.io/anvl/dev/reference/prim_convert.md)
for the underlying primitive.

## Examples

``` r
# the values are preserved, the data type changes
x <- nv_array(c(1L, 2L, 3L))
# For R inputs, we check compatability
nv_convert(x, dtype = "f32")
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
try(nv_convert(257L, dtype = "i8"))
#> Error : Value 257 cannot be converted to "i8" without overflow.
# For AnvlArrays, we wrap around
nv_convert(nv_scalar(257L), dtype = "i8")
#> AnvlArray
#>  1
#> [ CPUi8{} ] 
```
