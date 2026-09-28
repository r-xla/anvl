# Get the data type of an array

Returns the data type of an array (e.g. `f32`, `i64`).

## Usage

``` r
dtype(x, ...)
```

## Arguments

- x:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  \|
  [`GraphBox`](https://r-xla.github.io/anvl/dev/reference/GraphBox.md)
  \|
  [`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  An array. An R value has no data type of its own, so it is an error
  here; use
  [`peek_dtype()`](https://r-xla.github.io/anvl/dev/reference/peek_dtype.md)
  for the data type it would take.

- ...:

  Additional arguments passed to methods (unused).

## Value

([`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))

## Details

This is implemented via the generic
[`tengen::dtype()`](https://r-xla.github.io/tengen/reference/dtype.html).

## See also

[`tengen::dtype()`](https://r-xla.github.io/tengen/reference/dtype.html),
[`peek_dtype()`](https://r-xla.github.io/anvl/dev/reference/peek_dtype.md)

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
dtype(x)
#> <f32>
```
