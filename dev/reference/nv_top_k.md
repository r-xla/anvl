# Top-K Elements

Returns the `k` largest values along a dimension, sorted in decreasing
order.

## Usage

``` r
nv_top_k(operand, k, dim = NULL, with_indices = FALSE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Operand.

- k:

  (`integer(1)`)  
  Number of top elements to return. Must satisfy
  `1 <= k <= shape(operand)[dim]`.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to take the top `k`. If `NULL` (default), uses
  the last dimension.

- with_indices:

  (`logical(1)`)  
  If `FALSE` (default), returns just the top-`k` values. If `TRUE`,
  returns `list(values = ..., indices = ...)` where `indices` is the
  1-based position of each top-`k` value along `dim` (dtype `i32`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
(when `with_indices = FALSE`) or named list of two arrays (when
`with_indices = TRUE`). Output shape matches `operand` with `dim`
resized to `k`; values are sorted decreasing along `dim`.

## NaN handling

`NaN` ranks larger than any finite value (so it appears first in the
top-`k` output); `-NaN` ranks smaller. Unlike
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md),
the sign bit is not canonicalized.

## See also

[`prim_top_k()`](https://r-xla.github.io/anvl/dev/reference/prim_top_k.md)
for the underlying primitive,
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md).

## Examples

``` r
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
nv_top_k(x, k = 3L)
#> AnvlArray
#>  9
#>  6
#>  5
#> [ CPUf32{3} ] 
nv_top_k(x, k = 3L, with_indices = TRUE)
#> $values
#> AnvlArray
#>  9
#>  6
#>  5
#> [ CPUf32{3} ] 
#> 
#> $indices
#> AnvlArray
#>  6
#>  8
#>  5
#> [ CPUi32{3} ] 
#> 

m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
nv_top_k(m, k = 2L, dim = 2L)
#> AnvlArray
#>  5 3
#>  4 2
#> [ CPUf32{2,2} ] 
```
