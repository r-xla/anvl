# Primitive Max Reduction

Finds the maximum of array elements along the specified dimensions.

## Usage

``` r
prim_reduce_max(x, dims, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce over.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced dimensions from the output shape. If
  `TRUE`, the reduced dimensions are removed. If `FALSE`, the reduced
  dimensions are set to 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the shape is
that of `x` with `dims` removed. When `drop = FALSE`, the shape is that
of `x` with `dims` set to 1. It is ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
with
[`hlo_maximum()`](https://r-xla.github.io/stablehlo/reference/hlo_maximum.html)
as the reducer.

## See also

[`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
prim_reduce_max(x, dims = 1L)
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUi32{3} ] 
```
