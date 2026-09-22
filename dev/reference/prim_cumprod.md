# Primitive Cumulative Product

Cumulative product of array elements along a single axis. Output
position `j` along `axis` equals the product of input positions `1:j`.

A boolean input is accumulated with a logical AND, so the result is a
running AND.
[`nv_cumprod()`](https://r-xla.github.io/anvl/dev/reference/nv_cumprod.md)
multiplies zeroes and ones instead.

## Usage

``` r
prim_cumprod(x, axis)
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

## StableHLO

Lowers to
[`hlo_reduce_window()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce_window.html),
specified under
[reduce_window](https://openxla.org/stablehlo/spec#reduce_window). The
reducer is
[`hlo_multiply()`](https://r-xla.github.io/stablehlo/reference/hlo_multiply.html).

## See also

[`nv_cumprod()`](https://r-xla.github.io/anvl/dev/reference/nv_cumprod.md)

## Examples

``` r
# the accumulation keeps the input's data type
x <- nv_matrix(1:6, nrow = 2)
prim_cumprod(x, axis = 1L)
#> AnvlArray
#>   1  3  5
#>   2 12 30
#> [ CPUi32{2,3} ] 
```
