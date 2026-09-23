# Primitive Min Reduction

Finds the minimum of array elements along the specified axes.

## Usage

``` r
prim_min(x, axes, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes to reduce over. Negative values count from the end, i.e. `-1`
  refers to the last axis.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced axes: removed from the output shape if
  `TRUE`, set to 1 if `FALSE`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type. The shape is the input's with the reduced
axes removed (`drop = TRUE`) or set to 1 (`drop = FALSE`).

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html),
specified under [reduce](https://openxla.org/stablehlo/spec#reduce). The
reducer is
[`hlo_minimum()`](https://r-xla.github.io/stablehlo/reference/hlo_minimum.html).

## See also

[`nv_min()`](https://r-xla.github.io/anvl/dev/reference/nv_min.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
# reducing axis 1 removes it from the shape, and the data type is kept
prim_min(x, axes = 1L)
#> AnvlArray
#>  1
#>  3
#>  5
#> [ CPUi32{3} ] 

# drop = FALSE keeps the reduced axis at size 1 instead
prim_min(x, axes = 1L, drop = FALSE)
#> AnvlArray
#>  1 3 5
#> [ CPUi32{1,3} ] 

# reducing every axis gives a scalar
prim_min(x, axes = c(1L, 2L))
#> AnvlArray
#>  1
#> [ CPUi32{} ] 
```
