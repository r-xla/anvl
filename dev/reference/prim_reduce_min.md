# Primitive Min Reduction

Finds the minimum of array elements along the specified axes.

## Usage

``` r
prim_reduce_min(x, axes, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes to reduce over. Negative values count from the end, i.e. `-1`
  refers to the last axis.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced axes from the output shape. If `TRUE`, the
  reduced axes are removed. If `FALSE`, the reduced axes are set to 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the shape is
that of `x` with `axes` removed. When `drop = FALSE`, the shape is that
of `x` with `axes` set to 1. It is ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
with
[`hlo_minimum()`](https://r-xla.github.io/stablehlo/reference/hlo_minimum.html)
as the reducer.

## See also

[`nv_reduce_min()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_min.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
prim_reduce_min(x, axes = 1L)
#> AnvlArray
#>  1
#>  3
#>  5
#> [ CPUi32{3} ] 
```
