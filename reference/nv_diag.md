# Diagonal Matrix

Creates a diagonal matrix from a 1-D array.

## Usage

``` r
nv_diag(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  A 1-D array of length `n` whose elements become the diagonal entries.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
An `n x n` matrix with `x` on the diagonal and zeros elsewhere.

## Examples

``` r
nv_diag(nv_array(c(1, 2, 3)))
#> AnvlArray
#>  1 0 0
#>  0 2 0
#>  0 0 3
#> [ CPUf32{3,3} ] 
```
