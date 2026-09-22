# Outer Product

Computes the outer product of two 1-D arrays.

## Usage

``` r
nv_outer(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two 1-D arrays. Can be of any data type; the two are [promoted to a
  common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the operands' common data type and shape
`(length(lhs), length(rhs))`.

## Examples

``` r
# a length-3 and a length-2 vector give a 3x2 at their common data type
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5))
nv_outer(x, y)
#> AnvlArray
#>   4  5
#>   8 10
#>  12 15
#> [ CPUf32{3,2} ] 
```
