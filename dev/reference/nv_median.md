# Median

Computes the median over one or more axes. Equivalent to
`nv_quantile(x, 0.5, axes, drop, interpolation)`; for an even number of
reduced elements with the default `"linear"` interpolation, the average
of the two middle values is returned, matching base R's
[`median()`](https://rdrr.io/r/stats/median.html).

You can also use [`median()`](https://rdrr.io/r/stats/median.html)
directly on an
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
or [`AnvlBox`](https://r-xla.github.io/anvl/dev/reference/AnvlBox.md);
extra arguments (e.g. `interpolation`) are forwarded via `...`.

## Usage

``` r
nv_median(
  x,
  axes = NULL,
  drop = TRUE,
  interpolation = "linear",
  nan_rm = FALSE
)

# S3 method for class 'AnvlArray'
median(
  x,
  na.rm = FALSE,
  ...,
  axes = NULL,
  drop = TRUE,
  interpolation = "linear"
)
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

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Same shape as `x` with `axes` removed (or set to 1 if `drop = FALSE`).
The data type is that of `x`, or the default float for a non-float `x`.

## The [`median()`](https://rdrr.io/r/stats/median.html) generic

[`stats::median()`](https://rdrr.io/r/stats/median.html) reduces every
axis of a multi-axis array, and so does `nv_median()` by default, so the
two agree. Pass `axes` to reduce a subset instead. A non-float `x` is
computed at the default float, like base R returns a double.

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
m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
nv_median(m) # over every element
#> AnvlArray
#>  2.5000
#> [ CPUf32{} ] 
nv_median(m, axes = 2L) # one median per row
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
