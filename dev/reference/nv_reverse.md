# Reverse

Reverses the order of elements along the given axes, every axis by
default. You can also use [`rev()`](https://rdrr.io/r/base/rev.html),
which always reverses along every axis.

## Usage

``` r
nv_reverse(x, axes = NULL)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reverse. Negative values count from the end, i.e. `-1` refers
  to the last axis. If `NULL` (default), reverses along every axis.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
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
# the order along axis 1 is flipped
x <- nv_array(c(1, 2, 3, 4, 5))
nv_reverse(x)
#> AnvlArray
#>  5
#>  4
#>  3
#>  2
#>  1
#> [ CPUf32{5} ] 

m <- nv_matrix(1:6, nrow = 2)
nv_reverse(m) # every axis
#> AnvlArray
#>  6 4 2
#>  5 3 1
#> [ CPUi32{2,3} ] 
nv_reverse(m, axes = 2L) # columns only
#> AnvlArray
#>  5 3 1
#>  6 4 2
#> [ CPUi32{2,3} ] 
```
