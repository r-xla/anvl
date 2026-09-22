# Select Elements Along an Axis

Picks one or more elements along axis `axis` of `x`. Use this instead of
`[` or `nv_subset` when the index to select is provided
programmatically.

## Usage

``` r
nv_select(x, axis, index)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axis:

  (`integer(1)`)  
  Axis to index into. Negative values count from the end, i.e. `-1`
  refers to the last axis.

- index:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar or 1-D array of an integer data type, which it keeps – the
  index takes no part in `x`'s data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type. `axis` is dropped if `index` was scalar, and
otherwise resized to the number of selected elements.

## See also

[`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md)
for general subsetting,
[`prim_static_slice()`](https://r-xla.github.io/anvl/dev/reference/prim_static_slice.md).

## Examples

``` r
# a scalar index drops the axis, an index array keeps it
m <- nv_matrix(1:6, nrow = 2)
nv_select(m, axis = 2L, index = 2L)
#> AnvlArray
#>  3
#>  4
#> [ CPUi32{2} ] 
nv_select(m, axis = 1L, index = 1L)
#> AnvlArray
#>  1
#>  3
#>  5
#> [ CPUi32{3} ] 
nv_select(m, axis = 2L, index = array(c(1L, 3L)))
#> AnvlArray
#>  1 5
#>  2 6
#> [ CPUi32{2,2} ] 
```
