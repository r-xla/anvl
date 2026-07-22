# Select Elements Along a Dimension

Picks one or more elements along dimension `dim` of `x`. Use this
instead of `[` or `nv_subset` when the index to select is provided
programmatically.

## Usage

``` r
nv_select(x, dim, index)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- dim:

  (`integer(1)`)  
  Dimension to index into. Negative values count from the end, i.e. `-1`
  refers to the last dimension.

- index:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar or 1D arrayish input (integer).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Same data type as `x`. `dim` is dropped if `index` was scalar.

## See also

[`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md)
for general subsetting,
[`prim_static_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_static_slice.md).

## Examples

``` r
m <- nv_matrix(1:6, nrow = 2)
nv_select(m, dim = 2L, index = 2L)
#> AnvlArray
#>  3
#>  4
#> [ CPUi32{2} ] 
nv_select(m, dim = 1L, index = 1L)
#> AnvlArray
#>  1
#>  3
#>  5
#> [ CPUi32{3} ] 
nv_select(m, dim = 2L, index = array(c(1L, 3L)))
#> AnvlArray
#>  1 5
#>  2 6
#> [ CPUi32{2,2} ] 
```
