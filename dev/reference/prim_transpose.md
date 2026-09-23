# Primitive Transpose

Permutes the axes of an array.

## Usage

``` r
prim_transpose(x, perm)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- perm:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Specifies the new ordering of axes. Must be a permutation of
  `seq_len(naxes(x))`, the axis indices of `x`. Negative values count
  from the end, i.e. `-1` refers to the last axis.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type and shape `shape(x)[perm]`.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_transpose()`](https://r-xla.github.io/stablehlo/reference/hlo_transpose.html),
specified under
[transpose](https://openxla.org/stablehlo/spec#transpose).

## See also

[`nv_aperm()`](https://r-xla.github.io/anvl/dev/reference/nv_aperm.md),
[`t()`](https://rdrr.io/r/base/t.html)

## Examples

``` r
# the 2x3 becomes a 3x2, keeping its data type
x <- nv_matrix(1:6, nrow = 2)
prim_transpose(x, perm = c(2L, 1L))
#> AnvlArray
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
