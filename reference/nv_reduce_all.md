# All Reduction

Performs logical AND along the specified axes. Returns `TRUE` only if
all elements are `TRUE`.

## Usage

``` r
nv_reduce_all(x, axes = NULL, drop = TRUE)
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

[`prim_reduce_all()`](https://r-xla.github.io/anvl/reference/prim_reduce_all.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
nv_reduce_all(x)            # all axes -> scalar
#> AnvlArray
#>  0
#> [ CPUbool{} ] 
nv_reduce_all(x, axes = 1L)
#> AnvlArray
#>  0
#>  1
#> [ CPUbool{2} ] 
```
