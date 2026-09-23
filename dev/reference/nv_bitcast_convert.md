# Bitcast Conversion

Reinterprets the bits of an array as a different data type without
modifying the underlying data. If the target type is narrower, an extra
trailing axis is added; if wider, the last axis is consumed.

## Usage

``` r
nv_bitcast_convert(x, dtype)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type except `bool`. An R value materializes
  at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Any target data type except `bool`. One of the same bit width as the
  input's leaves the shape unchanged; a narrower one adds a *leading*
  axis holding the pieces; a wider one consumes the first axis, whose
  size must equal the ratio of the two widths. The pieces of one element
  are therefore adjacent in the column-major element order
  [`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
  reads, and a narrowing conversion lays the bytes out the way
  [`as_raw()`](https://r-xla.github.io/anvl/dev/reference/as_raw.md)
  writes them.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the given `dtype`, and the shape described under `dtype`.

## See also

[`prim_bitcast_convert()`](https://r-xla.github.io/anvl/dev/reference/prim_bitcast_convert.md),
which this is an alias of, and
[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)
for value-preserving type conversion.

## Examples

``` r
# the bits of one i32 reread as four i8, in a new trailing axis
x <- nv_array(1L, dtype = "i32")
nv_bitcast_convert(x, dtype = "i8")
#> AnvlArray
#>  1
#>  0
#>  0
#>  0
#> [ CPUi8{4,1} ] 
```
