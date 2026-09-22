# Primitive Broadcast

Broadcasts an array to a new shape by replicating the data along new or
size-1 axes.

## Usage

``` r
prim_broadcast_in_axes(x, shape, broadcast_axes)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. The size of each mapped axis must either match the size
  of the corresponding axis of `x`, or that axis of `x` must have size
  1.

- broadcast_axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Maps each axis of `x` to an axis of the output. Must have length equal
  to the number of axes of `x`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the same data type as the input and the given `shape`.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_broadcast_in_dim()`](https://r-xla.github.io/stablehlo/reference/hlo_broadcast_in_dim.html),
specified under
[broadcast_in_dim](https://openxla.org/stablehlo/spec#broadcast_in_dim).

## See also

[`nv_broadcast_to()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_to.md)

## Examples

``` r
# axis 1 of the input becomes axis 2 of the result, which repeats it
x <- nv_array(c(1, 2, 3))
prim_broadcast_in_axes(x, shape = c(2, 3), broadcast_axes = 2L)
#> AnvlArray
#>  1 2 3
#>  1 2 3
#> [ CPUf32{2,3} ] 
```
