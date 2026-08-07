# Get the number of axes of an array

Returns the number of axes (sometimes also refered to as rank) of an
array. Equivalent to `length(shape(x))`.

## Usage

``` r
naxes(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  An array-like object.

## Value

`integer(1)`

## See also

[`tengen::naxes()`](https://r-xla.github.io/tengen/reference/naxes.html)

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
naxes(x)
#> [1] 1
```
