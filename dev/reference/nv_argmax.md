# Index of the Maximum

Returns the index of the maximum value along an axis. Ties are broken by
returning the smallest index.

## Usage

``` r
nv_argmax(x, axis = NULL, drop = TRUE, nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to find the index. Negative values count from the
  end, i.e. `-1` refers to the last axis. If `NULL` (default), uses the
  last axis.

- drop:

  (`logical(1)`)  
  If `TRUE` (default) the reduced axis is removed; if `FALSE` it is kept
  with size 1.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) of
dtype `i32`  
Same shape as `x` with `axis` removed (or set to 1 if `drop = FALSE`).

## NaN handling

With `nan_rm = FALSE` (default), if any entry along the reduced axis is
`NaN`, the returned index points at the first such `NaN`. With
`nan_rm = TRUE`, `NaN` entries are skipped.

## See also

[`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md),
[`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md).

## Examples

``` r
nv_argmax(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#> AnvlArray
#>  6
#> [ CPUi32{} ] 
nv_argmax(nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE),
  axis = 2L
)
#> AnvlArray
#>  3
#>  2
#> [ CPUi32{2} ] 
nv_argmax(nv_array(c(1, NaN, 3)))
#> AnvlArray
#>  2
#> [ CPUi32{} ] 
nv_argmax(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#> AnvlArray
#>  3
#> [ CPUi32{} ] 
```
