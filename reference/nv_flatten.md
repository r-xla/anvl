# Flatte

Flattens an N-dimensional array into a 1-dimensional array. Fails with
scalar inputs.

## Usage

``` r
nv_flatten(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
1-D array.

## Examples

``` r
nv_flatten(matrix(1:4, nrow = 2))
#> AnvlArray
#>  1
#>  3
#>  2
#>  4
#> [ CPUi32?{4} ] 
```
