# Primitive All Reduction

Performs logical AND along the specified axes.

## Usage

``` r
prim_reduce_all(x, axes, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of boolean data type.

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
Boolean array. When `drop = TRUE`, the shape is that of `x` with `axes`
removed. When `drop = FALSE`, the shape is that of `x` with `axes` set
to 1.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
with
[`hlo_and()`](https://r-xla.github.io/stablehlo/reference/hlo_and.html)
as the reducer.

## See also

[`nv_reduce_all()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_all.md)

## Examples

``` r
x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
prim_reduce_all(x, axes = 1L)
#> AnvlArray
#>  0
#>  1
#> [ CPUbool{2} ] 
```
