# Get the Shape of an Array

Returns the shape of an array as an
[`integer()`](https://rdrr.io/r/base/integer.html) vector.

## Usage

``` r
shape(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  An array-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

([`integer()`](https://rdrr.io/r/base/integer.html))

## Details

An R value has a shape when it is arrayish: a length-1 vector is a
scalar with shape [`integer()`](https://rdrr.io/r/base/integer.html),
and an R array has its [`dim()`](https://rdrr.io/r/base/dim.html). Any
other R vector is an error; use
[`nv_array()`](https://r-xla.github.io/anvl/reference/AnvlArray.md) to
make it an array.

This is implemented via the generic
[`xlamisc::shape()`](https://r-xla.github.io/xlamisc/reference/shape.html).

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
shape(x)
#> [1] 4
shape(nv_array(1:6, shape = c(2, 3)))
#> [1] 2 3
# a length-1 R vector is a scalar
shape(1)
#> integer(0)
shape(TRUE)
#> integer(0)
# an R array has its `dim()`
shape(array(1:6, dim = c(2, 3)))
#> [1] 2 3
# any other R vector has no shape
try(shape(c(2, 3)))
#> Error in shape(c(2, 3)) : 
#>   `shape()` is undefined for a length-2 R vector.
#> ℹ Only a length-1 R value and an `array()` are arrayish; use `nv_array()` to
#>   make one an array.
```
