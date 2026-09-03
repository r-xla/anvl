# Primitive Cumulative Sum

Cumulative sum of array elements along a single axis. Output position
`j` along `axis` equals the sum of input positions `1:j`.

## Usage

``` r
prim_cumsum(x, axis)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- axis:

  (`integer(1)`)  
  Axis along which to accumulate. Negative values count from the end,
  i.e. `-1` refers to the last axis.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_reduce_window()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce_window.html)
with
[`hlo_add()`](https://r-xla.github.io/stablehlo/reference/hlo_add.html)
as the reducer.

## See also

[`nv_cumsum()`](https://r-xla.github.io/anvl/dev/reference/nv_cumsum.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
prim_cumsum(x, axis = 1L)
#> AnvlArray
#>   1  3  5
#>   3  7 11
#> [ CPUi32{2,3} ] 
```
