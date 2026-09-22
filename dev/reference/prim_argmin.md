# Primitive Index of the Minimum

Returns the index of the minimum value along a single axis. Ties are
broken by returning the smallest index.

## Usage

``` r
prim_argmin(x, axis, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axis:

  (`integer(1)`)  
  Axis along which to find the index of the minimum. Negative values
  count from the end, i.e. `-1` refers to the last axis.

- drop:

  (`logical(1)`)  
  If `TRUE` (default) the reduced axis is removed; if `FALSE` it is kept
  with size 1.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
regardless of the input's, and the input's shape with `axis` removed
(`drop = TRUE`) or set to 1 (`drop = FALSE`).

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html),
specified under [reduce](https://openxla.org/stablehlo/spec#reduce). The
reduction is variadic over `(values, indices)`, with a (value \< value
\| (value == value & idx \< idx)) selector.

## See also

[`prim_argmax()`](https://r-xla.github.io/anvl/dev/reference/prim_argmax.md),
[`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md)

## Examples

``` r
# the index comes out at the default integer data type
prim_argmin(nv_array(c(3, 1, 4, 1, 5)), axis = 1L)
#> AnvlArray
#>  2
#> [ CPUi32{} ] 
```
