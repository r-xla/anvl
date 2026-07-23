# Primitive Transpose

Permutes the axes of an array.

## Usage

``` r
prim_transpose(x, permutation)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Specifies the new ordering of axes. Must be a permutation of
  `seq_len(naxes(x))`, the axis indices of `x`. Negative values count
  from the end, i.e. `-1` refers to the last axis.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input and shape
`nv_shape(x)[permutation]`. It is ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_transpose()`](https://r-xla.github.io/stablehlo/reference/hlo_transpose.html).

## See also

[`nv_transpose()`](https://r-xla.github.io/anvl/dev/reference/nv_transpose.md),
[`t()`](https://rdrr.io/r/base/t.html)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
prim_transpose(x, permutation = c(2L, 1L))
#> AnvlArray
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
