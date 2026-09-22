# Index of the Maximum

Returns the index of the maximum value over one or more axes. Ties are
broken by returning the smallest index.

## Usage

``` r
nv_argmax(x, axes = NULL, drop = TRUE, nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reduce. Negative values count from the end, i.e. `-1` refers
  to the last axis. If `NULL` (default), reduces over all axes,
  returning a scalar.

- drop:

  (`logical(1)`)  
  Whether to drop reduced axes.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) of
the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))  
Same shape as `x` with `axes` removed (or set to 1 if `drop = FALSE`).

## Reducing several axes

`nv_argmax()` is the index to
[`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md)'s
value: called with the same `axes` and `drop`, it points at the element
whose value
[`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md)
returns. Reducing several axes ranks their elements together, and the
result indexes the row-major flattening of those axes – the order
[`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
produces – which is also the order ties are broken in.

## NaN handling

With `nan_rm = FALSE` (default), if any entry being reduced is `NaN`,
the returned index points at the first such `NaN`. With `nan_rm = TRUE`,
`NaN` entries are skipped.

## See also

[`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md),
[`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md).

## Examples

``` r
nv_argmax(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#> AnvlArray
#>  6
#> [ CPUi32{} ] 
m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
nv_argmax(m) # indexes the flattened matrix
#> AnvlArray
#>  3
#> [ CPUi32{} ] 
nv_argmax(m, axes = 2L) # one index per row
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
