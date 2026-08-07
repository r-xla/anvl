# Broadcast to Shape

Broadcasts an array to a target shape using NumPy-style broadcasting
rules.

## Usage

``` r
nv_broadcast_to(x, shape)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. Each existing axis must either match or be 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the given `shape` and the same data type as `x`.

## See also

[`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/reference/nv_broadcast_arrays.md),
[`nv_broadcast_scalars()`](https://r-xla.github.io/anvl/reference/nv_broadcast_scalars.md),
[`prim_broadcast_in_axes()`](https://r-xla.github.io/anvl/reference/prim_broadcast_in_axes.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
nv_broadcast_to(x, shape = c(2, 3))
#> AnvlArray
#>  1 2 3
#>  1 2 3
#> [ CPUf32{2,3} ] 
```
