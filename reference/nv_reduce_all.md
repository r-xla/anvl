# All Reduction

Performs logical AND along the specified dimensions. Returns `TRUE` only
if all elements are `TRUE`.

## Usage

``` r
nv_reduce_all(operand, dims = NULL, drop = TRUE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Dimensions to reduce. If `NULL` (default), reduces over all
  dimensions, returning a scalar.

- drop:

  (`logical(1)`)  
  Whether to drop reduced dimensions.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Boolean array. When `drop = TRUE`, the reduced dimensions are removed.
When `drop = FALSE`, the reduced dimensions are set to 1.

## See also

[`prim_reduce_all()`](https://r-xla.github.io/anvl/reference/prim_reduce_all.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
nv_reduce_all(x)            # all dims -> scalar
#> AnvlArray
#>  0
#> [ CPUbool{} ] 
nv_reduce_all(x, dims = 1L)
#> AnvlArray
#>  0
#>  1
#> [ CPUbool{2} ] 
```
