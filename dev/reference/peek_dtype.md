# Peek at a Data Type

The data type `x` would use if it was converted to an `AnvlArray`.
Relevant for R objects and their
[`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md)
trace-time analogon: for those it is the default of the backend in force
(see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)),
which the value has not committed to yet.

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
