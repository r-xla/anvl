# Peek at a Data Type

The data type `x` would use if was conveted to an `AnvlArray`. Relevant
for R objects and their
[`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md)
trace-time analogon.

## Usage

``` r
peek_dtype(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \|
  [`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  The value to ask about.

## Value

([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))

## See also

[`as_anvl_arrays()`](https://r-xla.github.io/anvl/dev/reference/as_anvl_array.md),
[RData](https://r-xla.github.io/anvl/dev/reference/RData.md),
[shape()](https://r-xla.github.io/tengen/reference/shape.html)

## Examples

``` r
peek_dtype(1.5)
#> <f32>
peek_dtype(1L)
#> <i32>
peek_dtype(nv_array(1:3, dtype = "i8"))
#> <i8>
```
