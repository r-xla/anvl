# Any Reduction

Performs logical OR along the specified axes. Returns `TRUE` if any
element is `TRUE`.

## Usage

``` r
nv_reduce_any(x, axes = NULL, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Axes to reduce. Negative values count from the end, i.e. `-1` refers
  to the last axis. If `NULL` (default), reduces over all axes,
  returning a scalar.

- drop:

  (`logical(1)`)  
  Whether to drop reduced axes.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Boolean array. When `drop = TRUE`, the reduced axes are removed. When
`drop = FALSE`, the reduced axes are set to 1.

## See also

[`prim_reduce_any()`](https://r-xla.github.io/anvl/reference/prim_reduce_any.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
nv_reduce_any(x)            # all axes -> scalar
#> AnvlArray
#>  1
#> [ CPUbool{} ] 
nv_reduce_any(x, axes = 1L)
#> AnvlArray
#>  1
#>  1
#> [ CPUbool{2} ] 
```
