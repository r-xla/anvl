# Get the data type of an array

Returns the data type of an array (e.g. `f32`, `i64`).

## Usage

``` r
dtype(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

([`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))

## Details

This is implemented via the generic
[`tengen::dtype()`](https://r-xla.github.io/tengen/reference/dtype.html).

## See also

[`tengen::dtype()`](https://r-xla.github.io/tengen/reference/dtype.html)

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
dtype(x)
#> <f32>
```
