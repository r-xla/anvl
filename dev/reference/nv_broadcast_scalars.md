# Broadcast Scalars to Common Shape

Broadcast scalar arrays to match the shape of non-scalar arrays. All
non-scalar arrays must have the same shape.

## Usage

``` r
nv_broadcast_scalars(...)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to broadcast.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
The inputs, each with its own data type and the common shape.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
# scalar 1 is broadcast to shape [3]
nv_broadcast_scalars(x, nv_scalar(1))
#> [[1]]
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1
#>  1
#>  1
#> [ CPUf32{3} ] 
#> 
```
