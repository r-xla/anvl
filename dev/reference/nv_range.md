# Range Reduction

The smallest and the largest element along the specified axes, stacked
along a new first axis. You can also use the
[`range()`](https://rdrr.io/r/base/range.html) generic.

## Usage

``` r
nv_range(x, axes = NULL, nan_rm = FALSE)

# S3 method for class 'AnvlArray'
range(..., na.rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reduce over. `NULL` (default) reduces over all of them, which
  makes the result a length-2 array like
  [`base::range()`](https://rdrr.io/r/base/range.html). Negative values
  count from the end.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in float inputs. If `FALSE` (default),
  `NaN` propagates. If `TRUE`, `NaN` values are skipped.

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to reduce, plus named arguments for
  [`nv_reduce_min()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_min.md)
  and
  [`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md)
  (e.g. `axes`), which are only accepted when there is a single array to
  reduce.

- na.rm:

  (`logical(1)`)  
  Forwarded to the `nan_rm` argument of
  [`nv_reduce_min()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_min.md)
  and
  [`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the same data type as `x` and the shape of the reduced array with a
leading axis of size 2 added: element 1 is the minimum, element 2 the
maximum.

## The [`range()`](https://rdrr.io/r/base/range.html) generic

[`range()`](https://rdrr.io/r/base/range.html) reduces over all axes
and, like [`base::range()`](https://rdrr.io/r/base/range.html), takes
several data arguments: `range(x, y)` is the range of both arrays.
Beyond base R, named arguments are passed on, so `range(x, axes = 1L)`
reduces a single axis – but only when `x` is the only data argument.

## See also

[`nv_reduce_min()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_min.md),
[`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md)

## Examples

``` r
nv_range(nv_array(c(3, 1, 4)))
#> AnvlArray
#>  1
#>  4
#> [ CPUf32{2} ] 
nv_range(nv_matrix(1:6, nrow = 2), axes = 1L)
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
```
