# Diagonal Matrix

Creates a diagonal matrix from a 1-D array.

## Usage

``` r
nv_diag(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, a 1-D array of length `n` whose elements become the
  diagonal entries. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type and shape `(n, n)`, with the input on the
diagonal and zeros elsewhere.

## Examples

``` r
# the vector becomes the diagonal of a 3x3 matrix
nv_diag(nv_array(c(1, 2, 3)))
#> AnvlArray
#>  1 0 0
#>  0 2 0
#>  0 0 3
#> [ CPUf32{3,3} ] 
```
