# Sort

Sorts an array along an axis.

You can also use [`sort()`](https://rdrr.io/r/base/sort.html) directly.

## Usage

``` r
nv_sort(x, axis = NULL, decreasing = FALSE, stable = FALSE)

# S3 method for class 'AnvlArray'
sort(x, decreasing = FALSE, ..., axis = NULL)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axis:

  (`integer(1)` \| `NULL`)  
  Axis along which to sort. Negative values count from the end, i.e.
  `-1` refers to the last axis. If `NULL` (default), the input is first
  flattened to a 1-D array, like
  [`base::sort()`](https://rdrr.io/r/base/sort.html).

- decreasing:

  (`logical(1)`)  
  If `TRUE`, sort in decreasing order.

- stable:

  (`logical(1)`)  
  If `TRUE`, the sort is stable: equal values keep their original
  relative order along `axis`. Default `FALSE`. Stability is only
  observable for floats when `-0` / `+0` or `-NaN` / `+NaN` are mixed
  (they compare equal under the total order used here); for distinct
  values the result is identical either way.

- ...:

  No additional arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## NaN handling

`NaN` values sort to the **end** (ascending) or **beginning**
(descending), regardless of sign. `+0` and `-0` compare equal.

## The [`sort()`](https://rdrr.io/r/base/sort.html) generic

Like [`base::sort()`](https://rdrr.io/r/base/sort.html), `nv_sort()`
with `axis = NULL` flattens a multi-axis array into one sorted vector,
so [`sort()`](https://rdrr.io/r/base/sort.html) on an anvl array agrees
with base R (the flatten order does not matter once the elements are
sorted). It differs in one respect: base R drops `NA` by default,
whereas `NaN` is kept and sorted to the end. Pass `axis` to sort each
slice along one axis instead, which keeps the shape.

## See also

[`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md)
for the underlying primitive,
[`nv_order()`](https://r-xla.github.io/anvl/dev/reference/nv_order.md),
[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
[`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md),
[`nv_which_max()`](https://r-xla.github.io/anvl/dev/reference/nv_which_max.md),
[`nv_which_min()`](https://r-xla.github.io/anvl/dev/reference/nv_which_min.md).

## Examples

``` r
# sorting moves elements, so the data type and shape stay
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
nv_sort(m) # one sorted vector, like base R
#> AnvlArray
#>  0
#>  1
#>  2
#>  3
#>  4
#>  5
#> [ CPUf32{6} ] 
nv_sort(m, axis = 2L) # each row sorted, shape kept
#> AnvlArray
#>  1 3 5
#>  0 2 4
#> [ CPUf32{2,3} ] 
```
