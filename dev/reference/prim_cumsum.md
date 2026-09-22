# Primitive Cumulative Sum

Cumulative sum of array elements along a single axis. Output position
`j` along `axis` equals the sum of input positions `1:j`.

A boolean input is accumulated with a logical OR, so the result is a
running OR rather than a running count.
[`nv_cumsum()`](https://r-xla.github.io/anvl/dev/reference/nv_cumsum.md)
counts instead.

## Usage

``` r
prim_cumsum(x, axis)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axis:

  (`integer(1)`)  
  Axis along which to accumulate. Negative values count from the end,
  i.e. `-1` refers to the last axis.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_reduce_window()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce_window.html),
specified under
[reduce_window](https://openxla.org/stablehlo/spec#reduce_window). The
reducer is
[`hlo_add()`](https://r-xla.github.io/stablehlo/reference/hlo_add.html).

## See also

[`nv_cumsum()`](https://r-xla.github.io/anvl/dev/reference/nv_cumsum.md)

## Examples

``` r
# the accumulation keeps the input's data type
x <- nv_matrix(1:6, nrow = 2)
prim_cumsum(x, axis = 1L)
#> AnvlArray
#>   1  3  5
#>   3  7 11
#> [ CPUi32{2,3} ] 
```
