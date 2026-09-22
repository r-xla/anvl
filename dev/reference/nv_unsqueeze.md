# Unsqueeze

Inserts an axis of size 1 at the specified position.

## Usage

``` r
nv_unsqueeze(x, axis)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axis:

  (`integer(1)`)  
  Position at which to insert the new axis. Valid positions range from 1
  to `naxes(x) + 1`. Negative values count from the end of the *result*,
  i.e. `-1` appends the new axis at the end.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type, with an extra axis of size 1 in its shape.

## See also

[`nv_squeeze()`](https://r-xla.github.io/anvl/dev/reference/nv_squeeze.md),
[`nv_reshape()`](https://r-xla.github.io/anvl/dev/reference/nv_reshape.md)

## Examples

``` r
# a size-1 axis is inserted, at the front or at the back
x <- nv_array(c(1, 2, 3))
nv_unsqueeze(x, axis = 1L)
#> AnvlArray
#>  1 2 3
#> [ CPUf32{1,3} ] 
nv_unsqueeze(x, axis = -1L)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3,1} ] 
```
