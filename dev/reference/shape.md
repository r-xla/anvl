# Get the shape of an array

Returns the shape of an array as an
[`integer()`](https://rdrr.io/r/base/integer.html) vector.

## Usage

``` r
shape(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

([`integer()`](https://rdrr.io/r/base/integer.html))

## Details

This is implemented via the generic
[`tengen::shape()`](https://r-xla.github.io/tengen/reference/shape.html).

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
shape(x)
#> [1] 4
```
