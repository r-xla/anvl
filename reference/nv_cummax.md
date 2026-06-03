# Cumulative Maximum

Running maximum, optionally along a single dimension.

## Usage

``` r
nv_cummax(operand, dim = NULL, with_indices = FALSE, nan_rm = FALSE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to accumulate. If `NULL` (default), the input is
  first flattened to a 1-D array, like
  [`base::cummax()`](https://rdrr.io/r/base/cumsum.html).

- with_indices:

  (`logical(1)`)  
  If `FALSE` (default), returns the running-maximum array. If `TRUE`,
  returns `list(values = ..., indices = ...)` where `indices` is the
  1-based index of the last occurrence of the running maximum at each
  position (dtype `i32`, matching torch). When `dim = NULL`, indices
  refer to the flattened input.

- nan_rm:

  (`logical(1)`)  
  How to handle `NaN` values in floating-point inputs. If `FALSE`
  (default), `NaN` propagates forward from its first occurrence. If
  `TRUE`, `NaN` is treated as the identity element of the cumulative op
  (`0` for sum, `1` for prod, `-Inf` / `+Inf` for max / min) and
  contributes nothing to the running value.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) (when
`with_indices = FALSE`) or named list of two arrays (when
`with_indices = TRUE`).

## Relation to base R

Both `nv_cummax()` (with `dim = NULL`) and
[`base::cummax()`](https://rdrr.io/r/base/cumsum.html) flatten a
multi-dimensional input to 1-D before accumulating, but the flatten
order differs: anvl arrays are row-major (C order), so the flattened
sequence iterates the last dim fastest, whereas base R uses column-major
(Fortran) order. The two agree on 1-D inputs.

## See also

[`prim_cummax()`](https://r-xla.github.io/anvl/reference/prim_cummax.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
nv_cummax(x)
#> AnvlArray
#>  3
#>  4
#>  5
#>  5
#>  5
#>  9
#> [ CPUf32{6} ] 
nv_cummax(x, dim = 1L)
#> AnvlArray
#>  3 4 5
#>  3 4 9
#> [ CPUf32{2,3} ] 
nv_cummax(x, dim = 1L, with_indices = TRUE)
#> $values
#> AnvlArray
#>  3 4 5
#>  3 4 9
#> [ CPUf32{2,3} ] 
#> 
#> $indices
#> AnvlArray
#>  1 1 1
#>  1 1 2
#> [ CPUi32{2,3} ] 
#> 
nv_cummax(nv_array(c(1, NaN, 3)))                # NaN propagates
#> AnvlArray
#>    1
#>  nan
#>  nan
#> [ CPUf32{3} ] 
nv_cummax(nv_array(c(1, NaN, 3)), nan_rm = TRUE) # NaN skipped
#> AnvlArray
#>  1
#>  1
#>  3
#> [ CPUf32{3} ] 
```
