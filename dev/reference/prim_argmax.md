# Primitive Argmax

Returns the index of the maximum value along a single axis. Ties are
broken by returning the smallest index.

## Usage

``` r
prim_argmax(x, axis, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- axis:

  (`integer(1)`)  
  Axis along which to find the index of the maximum. Negative values
  count from the end, i.e. `-1` refers to the last axis.

- drop:

  (`logical(1)`)  
  If `TRUE` (default) the reduced axis is removed; if `FALSE` it is kept
  with size 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) of
dtype `i32`  
Same shape as `x` with `axis` removed (or set to 1 if `drop = FALSE`).

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to a variadic
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
over `(values, indices)` with a (value \> value \| (value == value & idx
\< idx)) selector.

## See also

[`prim_argmin()`](https://r-xla.github.io/anvl/dev/reference/prim_argmin.md),
[`nv_argmax()`](https://r-xla.github.io/anvl/dev/reference/nv_argmax.md)

## Examples

``` r
prim_argmax(nv_array(c(3, 1, 4, 1, 5)), axis = 1L)
#> AnvlArray
#>  5
#> [ CPUi32{} ] 
```
