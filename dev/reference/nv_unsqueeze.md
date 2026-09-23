# Unsqueeze

Inserts axes of size 1 at the specified positions.

## Usage

``` r
nv_unsqueeze(x, axes)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Positions of the new axes in the *result*, which has
  `naxes(x) + length(axes)` axes. Negative values count from the end of
  the result, i.e. `-1` appends a new axis at the end.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type, with an extra axis of size 1 in its shape for each
of `axes`.

## See also

[`nv_squeeze()`](https://r-xla.github.io/anvl/dev/reference/nv_squeeze.md),
[`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md)

## Examples

``` r
# a size-1 axis is inserted, at the front or at the back
x <- nv_array(c(1, 2, 3))
nv_unsqueeze(x, axes = 1L)
#> AnvlArray
#>  1 2 3
#> [ CPUf32{1,3} ] 
nv_unsqueeze(x, axes = -1L)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3,1} ] 

# several at once
nv_unsqueeze(x, axes = c(1L, 3L))
#> AnvlArray
#> (1,.,.) =
#>  1
#>  2
#>  3
#> [ CPUf32{1,3,1} ] 
```
