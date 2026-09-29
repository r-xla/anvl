# Get the Number of Axes of an Array

Returns the number of axes (sometimes also referred to as rank) of an
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

(`integer(1)`)

## See also

[`xlamisc::naxes()`](https://r-xla.github.io/xlamisc/reference/naxes.html)

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
naxes(x)
#> [1] 1
```
