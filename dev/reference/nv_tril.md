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
  Input array.

- diagonal:

  (`integer(1)`)  
  Diagonal offset. `0` (default) is the main diagonal, positive values
  include diagonals above, negative values exclude diagonals below.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as `x`.

## See also

[`nv_triu()`](https://r-xla.github.io/anvl/dev/reference/nv_triu.md),
[`nv_lower_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_lower_tri.md)

## Examples

``` r
x <- nv_fill(1, c(3, 3))
nv_tril(x)
#> AnvlArray
#>  1 0 0
#>  1 1 0
#>  1 1 1
#> [ CPUf32{3,3} ] 
```
