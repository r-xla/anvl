# Max Reduction

Finds the maximum of array elements along the specified axes.

## Usage

``` r
nv_max(x, axes = NULL, drop = TRUE, nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reduce over. Negative values count from the end, i.e. `-1`
  refers to the last axis. If `NULL` (default), reduces over all axes.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced axes: removed from the output shape if
  `TRUE`, set to 1 if `FALSE`.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in float inputs. If `FALSE` (default),
  `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type. The shape is the input's with the reduced
axes removed (`drop = TRUE`) or set to 1 (`drop = FALSE`).

## The [`max()`](https://rdrr.io/r/base/Extremes.html) generic

[`max()`](https://rdrr.io/r/base/Extremes.html) reduces over all axes
and, like [`base::max()`](https://rdrr.io/r/base/Extremes.html), takes
several data arguments: `max(x, y)` is the largest element of both
arrays. `na.rm` becomes `nan_rm`. Beyond base R, named arguments are
passed on, so `max(x, axes = 1L)` reduces a single axis – but only when
`x` is the only data argument.

## See also

[`prim_max()`](https://r-xla.github.io/anvl/dev/reference/prim_max.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
# no axes given: reduce over all of them
nv_max(x)
#> AnvlArray
#>  6
#> [ CPUi32{} ] 

# reducing axis 1 removes it, drop = FALSE keeps it at size 1
nv_max(x, axes = 1L)
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUi32{3} ] 
nv_max(x, axes = 1L, drop = FALSE)
#> AnvlArray
#>  2 4 6
#> [ CPUi32{1,3} ] 

# negative axes count from the end
nv_max(x, axes = -1L)
#> AnvlArray
#>  5
#>  6
#> [ CPUi32{2} ] 

# NaN propagates unless nan_rm = TRUE
nv_max(nv_array(c(1, NaN, 3)))
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_max(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
```
