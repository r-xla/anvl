# Primitive Any Reduction

Performs logical OR along the specified dimensions.

## Usage

``` r
prim_reduce_any(operand, dims, drop = TRUE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of boolean data type.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce over.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced dimensions from the output shape. If
  `TRUE`, the reduced dimensions are removed. If `FALSE`, the reduced
  dimensions are set to 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Boolean array. Never ambiguous. When `drop = TRUE`, the shape is that of
`operand` with `dims` removed. When `drop = FALSE`, the shape is that of
`operand` with `dims` set to 1.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
with
[`stablehlo::hlo_or()`](https://r-xla.github.io/stablehlo/reference/hlo_or.html)
as the reducer.

## See also

[`nv_reduce_any()`](https://r-xla.github.io/anvl/reference/nv_reduce_any.md)

## Examples

``` r
x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
prim_reduce_any(x, dims = 1L)
#> AnvlArray
#>  1
#>  1
#> [ CPUbool{2} ] 
```
