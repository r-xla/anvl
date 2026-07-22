# Quantile

Computes the `probs` quantile(s) of an array along a dimension.

`probs` follows the same scalar-vs-array convention as
[`nv_select()`](https://r-xla.github.io/anvl/dev/reference/nv_select.md)'s
`index`:

- a length-1 numeric (e.g. `0.5`) treats `probs` as scalar — the output
  has `dim` removed, like a reduction;

- a 1-D R array (e.g. `array(c(0.25, 0.5, 0.75))`) prepends a leading
  dimension of size `length(probs)`.

Plain length-K (K \> 1) vectors are rejected; wrap with
[`array()`](https://rdrr.io/r/base/array.html) to make the array intent
explicit.

## Usage

``` r
nv_quantile(x, probs, dim = NULL, interpolation = "linear", nan_rm = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- probs:

  (`numeric(1)` \| 1-D `array`)  
  One or more probabilities in `[0, 1]`. Either a length-1 numeric
  (scalar; `dim` is dropped) or a 1-D `array` (a leading dim of size
  `length(probs)` is prepended). Plain length-K (K \> 1) vectors are
  rejected — wrap with [`array()`](https://rdrr.io/r/base/array.html).

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to compute the quantile. If `NULL` (default),
  uses the last dimension.

- interpolation:

  (`character(1)`)  
  One of `"linear"` (default), `"lower"`, `"higher"`, `"nearest"`,
  `"midpoint"`. See "Interpolation modes".

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates. If `TRUE`, `NaN` values are skipped.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
For scalar `probs`: same shape as `x` with `dim` removed. For array
`probs`: a **leading** dimension of size `length(probs)` is prepended.

## Interpolation modes

Let `h = (n - 1) * q` be the 0-based fractional index for an axis of
length `n` and probability `q`, with `lo = floor(h)`, `hi = ceil(h)`,
`frac = h - lo`. Then:

- `"linear"` (default): `(1 - frac) * sorted[lo] + frac * sorted[hi]`.

- `"lower"`: `sorted[lo]` — the lower bracket of `linear`.

- `"higher"`: `sorted[hi]` — the upper bracket of `linear`.

- `"nearest"`: `sorted[lo]` if `frac < 0.5` else `sorted[hi]`.

- `"midpoint"`: `(sorted[lo] + sorted[hi]) / 2`.

## See also

[`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md),
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md).

## Examples

``` r
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
nv_quantile(nv_array(c(1, NaN, 3, 5)), 0.5)
#> AnvlArray
#>  nan
#> [ CPUf32{} ] 
nv_quantile(nv_array(c(1, NaN, 3, 5)), 0.5, nan_rm = TRUE)
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
```
