# Upper Triangular Matrix

Returns the upper triangular part of a 2-D array, setting elements below
the specified diagonal to zero.

## Usage

``` r
nv_triu(operand, diagonal = 0L)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Operand.

- diagonal:

  (`integer(1)`)  
  Diagonal offset. `0` (default) is the main diagonal, positive values
  exclude diagonals above, negative values include diagonals below.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as `operand`.

## See also

[`nv_tril()`](https://r-xla.github.io/anvl/dev/reference/nv_tril.md),
[`nv_upper_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_upper_tri.md)

## Examples

``` r
x <- nv_fill(1, c(3, 3))
nv_triu(x)
#> AnvlArray
#>  1 1 1
#>  0 1 1
#>  0 0 1
#> [ CPUf32{3,3} ] 
```
