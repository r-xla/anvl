# Get the Axes of an Array

Returns the axis indices of an array, i.e. `seq_len(naxes(x))`.

## Usage

``` r
axes(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  An array-like object.

## Value

([`integer()`](https://rdrr.io/r/base/integer.html))

## See also

[`naxes()`](https://r-xla.github.io/anvl/reference/naxes.md),
[`shape()`](https://r-xla.github.io/anvl/reference/shape.md)

## Examples

``` r
x <- nv_array(1:6, shape = c(2, 3))
axes(x)
#> [1] 1 2
```
