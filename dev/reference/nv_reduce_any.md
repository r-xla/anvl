# Any Reduction

Performs logical OR along the specified dimensions. Returns `TRUE` if
any element is `TRUE`.

## Usage

``` r
nv_reduce_any(x, dims = NULL, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Dimensions to reduce. If `NULL` (default), reduces over all
  dimensions, returning a scalar.

- drop:

  (`logical(1)`)  
  Whether to drop reduced dimensions.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Boolean array. When `drop = TRUE`, the reduced dimensions are removed.
When `drop = FALSE`, the reduced dimensions are set to 1.

## See also

[`prim_reduce_any()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_any.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
nv_reduce_any(x)            # all dims -> scalar
#> AnvlArray
#>  1
#> [ CPUbool{} ] 
nv_reduce_any(x, dims = 1L)
#> AnvlArray
#>  1
#>  1
#> [ CPUbool{2} ] 
```
