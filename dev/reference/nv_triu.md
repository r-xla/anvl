# Upper Triangular Matrix

Returns the upper triangular part of a 2-D array, setting elements below
the specified diagonal to zero.

## Usage

``` r
nv_triu(x, diagonal = 0L)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with exactly 2 axes. Can be any data type. An R value
  materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- diagonal:

  (`integer(1)`)  
  Diagonal offset: the kept region is `col - row >= diagonal`. `0`
  (default) keeps the main diagonal and everything above it, a negative
  value keeps that many diagonals below it as well, and a positive one
  drops the main diagonal and `diagonal - 1` above it.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the same shape and data type as `x`.

## See also

[`nv_tril()`](https://r-xla.github.io/anvl/dev/reference/nv_tril.md),
[`nv_upper_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_upper_tri.md)

## Examples

``` r
# elements below the main diagonal become zero
x <- nv_fill(1, c(3, 3))
nv_triu(x)
#> AnvlArray
#>  1 1 1
#>  0 1 1
#>  0 0 1
#> [ CPUf32{3,3} ] 
```
