# Flatte

Flattens an N-dimensional array into a 1-dimensional array. Fails with
scalar inputs.

## Usage

``` r
nv_flatten(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
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
