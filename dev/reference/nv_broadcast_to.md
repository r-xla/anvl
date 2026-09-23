# Broadcast to Shape

Broadcasts an array to a target shape, aligning the array's axes with
the leading axes of `shape`, so that a vector fills a column. See
[`nv_broadcast_arrays`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_arrays.md)
for more information.

## Usage

``` r
nv_broadcast_to(x, shape)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. The input's axes are matched against its leading axes,
  and each must either match or be 1; trailing axes are added.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the given `shape` and the same data type as `x`.

## See also

[`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_arrays.md),
[`nv_broadcast_scalars()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md),
[`prim_broadcast_in_axes()`](https://r-xla.github.io/anvl/dev/reference/prim_broadcast_in_axes.md)
for the underlying primitive.

## Examples

``` r
# the vector fills a column and is repeated along the new trailing axis
x <- nv_array(c(1, 2, 3))
nv_broadcast_to(x, shape = c(3, 2))
#> AnvlArray
#>  1 1
#>  2 2
#>  3 3
#> [ CPUf32{3,2} ] 
```
