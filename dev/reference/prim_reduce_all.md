# Primitive All Reduction

Performs logical AND along the specified axes.

## Usage

``` r
prim_reduce_all(x, axes, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Must be a boolean or an R logical.

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
Has the boolean data type. The shape is the input's with the reduced
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
[`hlo_and()`](https://r-xla.github.io/stablehlo/reference/hlo_and.html).

## See also

[`nv_reduce_all()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_all.md)

## Examples

``` r
x <- nv_matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2)
# TRUE where every element is TRUE along axis 1
prim_reduce_all(x, axes = 1L)
#> AnvlArray
#>  0
#>  1
#> [ CPUbool{2} ] 

# drop = FALSE keeps the reduced axis at size 1 instead
prim_reduce_all(x, axes = 1L, drop = FALSE)
#> AnvlArray
#>  0 1
#> [ CPUbool{1,2} ] 
```
