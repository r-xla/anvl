# Median

Computes the median along an axis. Equivalent to
`nv_quantile(x, 0.5, axis, interpolation)`; for an even-length axis with
the default `"linear"` interpolation, the average of the two middle
values is returned, matching base R's
[`median()`](https://rdrr.io/r/stats/median.html).

You can also use [`median()`](https://rdrr.io/r/stats/median.html)
directly on an
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
or [`AnvlBox`](https://r-xla.github.io/anvl/dev/reference/AnvlBox.md);
extra arguments (e.g. `interpolation`) are forwarded via `...`.

## Usage

``` r
nv_median(x, axis = NULL, interpolation = "linear", nan_rm = FALSE)

# S3 method for class 'AnvlArray'
median(x, na.rm = FALSE, ..., axis = NULL, interpolation = "linear")
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to compute the median. Negative values count from the
  end, i.e. `-1` refers to the last axis. If `NULL` (default), uses the
  last axis.

- interpolation:

  (`character(1)`)  
  Forwarded to
  [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md).
  One of `"linear"` (default), `"lower"`, `"higher"`, `"nearest"`,
  `"midpoint"`.

- nan_rm:

  (`logical(1)`)  
  Forwarded to
  [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md).
  See its documentation for details.

- na.rm:

  Forwarded to `nv_median()`'s `nan_rm` argument.

- ...:

  No additional arguments.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Same shape as `x` with `axis` removed. The data type is that of `x`, or
the default float for a non-float `x`.

## The [`median()`](https://rdrr.io/r/stats/median.html) generic

[`stats::median()`](https://rdrr.io/r/stats/median.html) flattens a
multi-axis array, while `nv_median()` (and
[`median()`](https://rdrr.io/r/stats/median.html) on an anvl array)
reduces a single axis, the last one by default. Pass `axis` explicitly,
or flatten first with
[`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md),
to say which you mean. A non-float `x` is computed at the default float,
like base R returns a double.

## See also

[`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md),
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md),
[`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md).

## Examples

``` r
nv_median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#> AnvlArray
#>  3.5000
#> [ CPUf32{} ] 
median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#> AnvlArray
#>  3.5000
#> [ CPUf32{} ] 
nv_median(nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE),
  axis = 2L
)
#> AnvlArray
#>  3
#>  2
#> [ CPUf32{2} ] 
# forwards through the S3 generic via `...`
median(nv_array(c(1, 2, 3, 4)), interpolation = "lower")
#> AnvlArray
#>  2
#> [ CPUf32{} ] 
nv_median(nv_array(c(1, NaN, 3, 5)))
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_median(nv_array(c(1, NaN, 3, 5)), nan_rm = TRUE)
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
```
