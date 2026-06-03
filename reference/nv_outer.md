# Outer Product

Computes the outer product of two 1-D arrays.

## Usage

``` r
nv_outer(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  1-D arrays.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
A 2-D array of shape `(length(lhs), length(rhs))`.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5))
nv_outer(x, y)
#> AnvlArray
#>   4  5
#>   8 10
#>  12 15
#> [ CPUf32{3,2} ] 
```
