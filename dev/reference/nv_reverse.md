# Reverse

Reverses the order of elements along specified axes. You can also use
[`rev()`](https://rdrr.io/r/base/rev.html), which reverses along every
axis.

## Usage

``` r
nv_reverse(x, axes)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes to reverse. Negative values count from the end, i.e. `-1` refers
  to the last axis.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as `x`.

## The [`rev()`](https://rdrr.io/r/base/rev.html) generic

[`rev()`](https://rdrr.io/r/base/rev.html) reverses along every axis,
which puts the elements in the same order as
[`base::rev()`](https://rdrr.io/r/base/rev.html) does (base R flattens
the array to a vector first, whereas
[`rev()`](https://rdrr.io/r/base/rev.html) on an anvl array keeps the
shape).

## See also

[`prim_reverse()`](https://r-xla.github.io/anvl/dev/reference/prim_reverse.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3, 4, 5))
nv_reverse(x, axes = 1L)
#> AnvlArray
#>  5
#>  4
#>  3
#>  2
#>  1
#> [ CPUf32{5} ] 
```
