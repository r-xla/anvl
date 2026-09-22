# Primitive Cumulative Maximum

Running maximum of array elements along a single axis along with the
index of the last occurrence of the running maximum. At output position
`j`, the values output is `max(input[1:j])` and the indices output is
the largest `i` in `1:j` with `input[i] == values[j]` (last-occurrence
tiebreak).

## Usage

``` r
prim_cummax(x, axis)
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

(named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `values`, the running maximum at the input's data type, and
`indices`, the running argmax at the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
Both have the input's shape.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_reduce_window()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce_window.html),
specified under
[reduce_window](https://openxla.org/stablehlo/spec#reduce_window). The
window is variadic over `(values, iota)`, so the index of the running
extremum is carried alongside it.

## See also

[`nv_cummax()`](https://r-xla.github.io/anvl/dev/reference/nv_cummax.md)

## Examples

``` r
# `values` keeps the input's data type, `indices` is the default integer
x <- nv_matrix(c(3, 1, 4, 1, 5, 9), nrow = 2)
prim_cummax(x, axis = 1L)
#> $values
#> AnvlArray
#>  3 4 5
#>  3 4 9
#> [ CPUf32{2,3} ] 
#> 
#> $indices
#> AnvlArray
#>  1 1 1
#>  1 1 2
#> [ CPUi32{2,3} ] 
#> 
```
