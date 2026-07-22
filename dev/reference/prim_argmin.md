# Primitive Argmin

Returns the index of the minimum value along a single dimension. Ties
are broken by returning the smallest index.

## Usage

``` r
prim_argmin(x, dim, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- dim:

  (`integer(1)`)  
  Dimension along which to find the index of the maximum.

- drop:

  (`logical(1)`)  
  If `TRUE` (default) the reduced dimension is removed; if `FALSE` it is
  kept with size 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) of
dtype `i32`  
Same shape as `x` with `dim` removed (or set to 1 if `drop = FALSE`).

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to a variadic
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
over `(values, indices)` with a (value \< value \| (value == value & idx
\< idx)) selector.

## See also

[`prim_argmax()`](https://r-xla.github.io/anvl/dev/reference/prim_argmax.md),
[`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md)

## Examples

``` r
prim_argmin(nv_array(c(3, 1, 4, 1, 5)), dim = 1L)
#> AnvlArray
#>  2
#> [ CPUi32{} ] 
```
