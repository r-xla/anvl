# Primitive Cumulative Minimum

Running minimum of array elements along a single axis along with the
index of the last occurrence of the running minimum. At output position
`j`, the values output is `min(input[1:j])` and the indices output is
the largest `i` in `1:j` with `input[i] == values[j]` (last-occurrence
tiebreak).

## Usage

``` r
prim_cummin(x, axis)
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

`list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
values:  
The running minimum (same dtype as `x`) and the running argmin (dtype
`i32`, 1-based). Both have the same shape as `x`.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to a variadic
[`hlo_reduce_window()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce_window.html)
over `(values, iota)`.

## See also

[`nv_cummin()`](https://r-xla.github.io/anvl/dev/reference/nv_cummin.md)

## Examples

``` r
x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
prim_cummin(x, axis = 1L)
#> [[1]]
#> AnvlArray
#>  3 4 5
#>  1 1 5
#> [ CPUf32{2,3} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1 1 1
#>  2 2 1
#> [ CPUi32{2,3} ] 
#> 
```
