# Lower Triangular Matrix

Returns the lower triangular part of a 2-D array, setting elements above
the specified diagonal to zero.

## Usage

``` r
nv_tril(x, diagonal = 0L)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with exactly 2 axes. Can be any data type. An R value
  materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- diagonal:

  (`integer(1)`)  
  Diagonal offset: the kept region is `col - row <= diagonal`. `0`
  (default) keeps the main diagonal and everything below it, a positive
  value keeps that many diagonals above it as well, and a negative one
  drops the main diagonal and `-diagonal - 1` below it.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the same shape and data type as `x`.

## See also

[`nv_triu()`](https://r-xla.github.io/anvl/dev/reference/nv_triu.md),
[`nv_lower_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_lower_tri.md)

## Examples

``` r
# elements above the main diagonal become zero
x <- nv_fill(1, c(3, 3))
nv_tril(x)
#> AnvlArray
#>  1 0 0
#>  1 1 0
#>  1 1 1
#> [ CPUf32{3,3} ] 
```
