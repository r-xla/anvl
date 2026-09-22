# Quantile

Computes the `probs` quantile(s) of an array over one or more axes.

`probs` follows the same scalar-vs-array convention as
[`nv_select()`](https://r-xla.github.io/anvl/dev/reference/nv_select.md)'s
`index`:

- a length-1 numeric (e.g. `0.5`) treats `probs` as scalar — the result
  is the reduction alone;

- a 1-D R array (e.g. `array(c(0.25, 0.5, 0.75))`) prepends a leading
  axis of size `length(probs)`.

Plain length-K (K \> 1) vectors are rejected; wrap with
[`array()`](https://rdrr.io/r/base/array.html) to make the array intent
explicit.

A quantile generally falls between two elements, so a non-float `x` is
computed (and returned) at the default float data type.

## Usage

``` r
nv_quantile(
  x,
  probs,
  axes = NULL,
  drop = TRUE,
  interpolation = "linear",
  nan_rm = FALSE
)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- probs:

  (`numeric(1)` \| 1-D `array`)  
  One or more probabilities in `[0, 1]`. Either a length-1 numeric
  (scalar) or a 1-D `array` (a leading axis of size `length(probs)` is
  prepended). Plain length-K (K \> 1) vectors are rejected — wrap with
  [`array()`](https://rdrr.io/r/base/array.html).

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
  One of `"linear"` (default), `"lower"`, `"higher"`, `"nearest"`,
  `"midpoint"`. See "Interpolation modes".

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in float inputs. If `FALSE` (default),
  `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Same shape as `x` with `axes` removed (or set to 1 if `drop = FALSE`).
For array `probs`, a **leading** axis of size `length(probs)` is
prepended. The data type is that of `x`, or the default float for a
non-float `x`.

## Interpolation modes

For `n` reduced elements and a probability `q`, let
`h = 1 + (n - 1) * q` be the position `q` falls at in the sorted values,
with `lo = floor(h)`, `hi = ceiling(h)` and `frac = h - lo`. Then,
writing `sorted` for the reduced values in sorted order:

- `"linear"` (default): `(1 - frac) * sorted[lo] + frac * sorted[hi]`.

- `"lower"`: `sorted[lo]` — the lower bracket of `linear`.

- `"higher"`: `sorted[hi]` — the upper bracket of `linear`.

- `"nearest"`: `sorted[lo]` if `frac < 0.5` else `sorted[hi]`.

- `"midpoint"`: `(sorted[lo] + sorted[hi]) / 2`.

Reducing several axes at once ranks all of their elements together, so
`nv_quantile(x, q, axes = c(1, 2))` equals
`nv_quantile(nv_flatten(x), q)` for a matrix `x`.

## See also

[`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md),
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md).

## Examples

``` r
# a float result even for an integer input, since it interpolates
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
nv_quantile(x, 0.5) # = nv_median(x)
#> AnvlArray
#>  3.5000
#> [ CPUf32{} ] 
nv_quantile(x, array(c(0.25, 0.5, 0.75)))
#> AnvlArray
#>  1.7500
#>  3.5000
#>  5.2500
#> [ CPUf32{3} ] 
nv_quantile(x, 0.5, interpolation = "lower")
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
nv_quantile(m, 0.5) # over every element
#> AnvlArray
#>  2.5000
#> [ CPUf32{} ] 
nv_quantile(m, 0.5, axes = 2L) # one quantile per row
#> AnvlArray
#>  3
#>  2
#> [ CPUf32{2} ] 
nv_quantile(nv_array(c(1, NaN, 3, 5)), 0.5)
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_quantile(nv_array(c(1, NaN, 3, 5)), 0.5, nan_rm = TRUE)
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
```
