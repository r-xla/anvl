# Sum Reduction

Sums array elements along the specified axes. A boolean array is
counted, like [`base::sum()`](https://rdrr.io/r/base/sum.html) does.

## Usage

``` r
nv_sum(x, axes = NULL, drop = TRUE, nan_rm = FALSE)
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
Has the input's data type, except a boolean input, which is accumulated
at the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
The shape is the input's with the reduced axes removed (`drop = TRUE`)
or set to 1 (`drop = FALSE`).

## The [`sum()`](https://rdrr.io/r/base/sum.html) generic

[`sum()`](https://rdrr.io/r/base/sum.html) reduces over all axes and,
like [`base::sum()`](https://rdrr.io/r/base/sum.html), takes several
data arguments: `sum(x, y)` is the sum of both arrays. `na.rm` becomes
`nan_rm`. Beyond base R, named arguments are passed on, so
`sum(x, axes = 1L)` reduces a single axis – but only when `x` is the
only data argument.

## See also

[`prim_sum()`](https://r-xla.github.io/anvl/dev/reference/prim_sum.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
# no axes given: reduce over all of them
nv_sum(x)
#> AnvlArray
#>  21
#> [ CPUi32{} ] 

# reducing axis 1 removes it, drop = FALSE keeps it at size 1
nv_sum(x, axes = 1L)
#> AnvlArray
#>   3
#>   7
#>  11
#> [ CPUi32{3} ] 
nv_sum(x, axes = 1L, drop = FALSE)
#> AnvlArray
#>   3  7 11
#> [ CPUi32{1,3} ] 

# negative axes count from the end
nv_sum(x, axes = -1L)
#> AnvlArray
#>   9
#>  12
#> [ CPUi32{2} ] 

# NaN propagates unless nan_rm = TRUE
nv_sum(nv_array(c(1, NaN, 3)))
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_sum(nv_array(c(1, NaN, 3)), nan_rm = TRUE)
#> AnvlArray
#>  4
#> [ CPUf32{} ] 
nv_sum(nv_array(c(TRUE, FALSE, TRUE))) # counts: 2
#> AnvlArray
#>  2
#> [ CPUi32{} ] 
```
