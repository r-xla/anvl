# Sort

Sorts an array along a dimension.

You can also use [`sort()`](https://rdrr.io/r/base/sort.html) directly.

## Usage

``` r
nv_sort(operand, dim = NULL, decreasing = FALSE, stable = FALSE)

# S3 method for class 'AnvlArray'
sort(x, decreasing = FALSE, ..., dim = NULL)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Operand.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to sort. If `NULL` (default), uses the last
  dimension.

- decreasing:

  (`logical(1)`)  
  If `TRUE`, sort in decreasing order.

- stable:

  (`logical(1)`)  
  If `TRUE`, the sort is stable: equal values keep their original
  relative order along `dim`. Default `FALSE`. Stability is only
  observable for floats when `-0` / `+0` or `-NaN` / `+NaN` are mixed
  (they compare equal under the total order used here); for distinct
  values the result is identical either way.

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Same as `operand`; this is the name used by the base R S3 generic.

- ...:

  No additional arguments.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Same shape and data type as `operand`.

## NaN handling

`NaN` values sort to the **end** (ascending) or **beginning**
(descending), regardless of sign. `+0` and `-0` compare equal.

## See also

[`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md)
for the underlying primitive,
[`nv_argsort()`](https://r-xla.github.io/anvl/dev/reference/nv_argsort.md),
[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
[`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md),
[`nv_argmax()`](https://r-xla.github.io/anvl/dev/reference/nv_argmax.md),
[`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md).

## Examples

``` r
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
nv_sort(x)
#> AnvlArray
#>  1
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#>  9
#> [ CPUf32{8} ] 
sort(x) # via the S3 generic
#> AnvlArray
#>  1
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#>  9
#> [ CPUf32{8} ] 
nv_sort(x, decreasing = TRUE)
#> AnvlArray
#>  9
#>  6
#>  5
#>  4
#>  3
#>  2
#>  1
#>  1
#> [ CPUf32{8} ] 

m <- nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)
nv_sort(m, dim = 2L)
#> AnvlArray
#>  1 3 5
#>  0 2 4
#> [ CPUf32{2,3} ] 
```
