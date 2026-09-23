# Broadcast Arrays to a Common Shape

Broadcasts arrays to a common shape, aligning their axes from the first
one, so that a vector meets a matrix as a column.

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

1.  If the arrays have different numbers of axes, append size-1 axes to
    the shorter shape, so axis 1 meets axis 1. A length-`n` vector
    therefore lines up with the rows of an `n` by `m` matrix and is
    replicated across its columns. NumPy prepends instead.

2.  For each axis: if the sizes match, keep them; if one is 1, expand it
    to the other's size; otherwise raise an error.

## Relation to base R

Base R has no broadcasting between arrays –
`matrix(1, 3, 3) + matrix(1, 1, 3)` is a "non-conformable arrays" error.
It does recycle a *vector* over a matrix, though, and for a vector as
long as the first axis that lands on exactly this broadcast, which is
why a vector meets a matrix as a column in both. The two part ways once
the lengths stop lining up: `matrix(1, 2, 3) + c(1, 2, 3)` recycles on
regardless, where the matching broadcast is an error.

The deviation from NumPy's broadcasting rules is still motivated by
keeping anvl's behaviour similar to base R in spirit, see the examples
for more.

## See also

[`nv_broadcast_scalars()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md),
[`nv_broadcast_to()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_to.md)

## Examples

``` r
# interpreting vectors as columns
# base R:
x1 <- c(1, 2)
m1 <- array(1, dim = c(2, 2))
x1 + m1
#>      [,1] [,2]
#> [1,]    2    2
#> [2,]    3    3
# anvl:
args <- nv_broadcast_arrays(nv_array(x1), nv_array(m1))
print(args)
#> [[1]]
#> AnvlArray
#>  1 1
#>  2 2
#> [ CPUf32{2,2} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1 1
#>  1 1
#> [ CPUf32{2,2} ] 
#> 
args[[1]] + args[[2]]
#> AnvlArray
#>  2 2
#>  3 3
#> [ CPUf32{2,2} ] 


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
