# Broadcast Arrays to a Common Shape

Broadcasts arrays to a common shape using NumPy-style broadcasting
rules.

## Usage

``` r
nv_broadcast_arrays(...)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to broadcast.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
The inputs, each with its own data type and the common shape.

## Broadcasting Rules

1.  If the arrays have different numbers of axes, prepend size-1 axes to
    the shorter shape.

2.  For each axis: if the sizes match, keep them; if one is 1, expand it
    to the other's size; otherwise raise an error.

## See also

[`nv_broadcast_scalars()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md),
[`nv_broadcast_to()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_to.md)

## Examples

``` r
# the length-3 vector is stretched to the matrix's shape
x1 <- nv_matrix(1:6, nrow = 2)
x2 <- nv_array(c(10, 20, 30))
nv_broadcast_arrays(x1, x2)
#> [[1]]
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
#> 
#> [[2]]
#> AnvlArray
#>  10 20 30
#>  10 20 30
#> [ CPUf32{2,3} ] 
#> 

# axes of size 1 are expanded to the other operand's size
y1 <- nv_array(1:3, shape = c(1, 3))
y2 <- nv_array(1:3, shape = c(3, 1))
nv_broadcast_arrays(y1, y2)
#> [[1]]
#> AnvlArray
#>  1 2 3
#>  1 2 3
#>  1 2 3
#> [ CPUi32{3,3} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1 1 1
#>  2 2 2
#>  3 3 3
#> [ CPUi32{3,3} ] 
#> 
```
