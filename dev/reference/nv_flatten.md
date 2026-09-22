# Flatten

Flattens an array of any rank into an array with a single axis, reading
the elements in row-major order (the last axis fastest), as
[`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md)
does.

## Usage

``` r
nv_flatten(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type, and one axis holding all of its elements – so
a scalar, which has none, becomes a length-1 vector.

## Differences from base R

Note that row-major order is used, which differs from R's column-major
order.

## Examples

``` r
# the 2x2 matrix becomes a length-4 vector
x <- matrix(1:4, nrow = 2)
# flatten in row-major
nv_flatten(x)
#> AnvlArray
#>  1
#>  3
#>  2
#>  4
#> [ CPUi32{4} ] 
# differs from R's col-major encoding:
c(x)
#> [1] 1 2 3 4
```
