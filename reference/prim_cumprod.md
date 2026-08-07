# Primitive Cumulative Product

Cumulative product of array elements along a single axis. Output
position `j` along `axis` equals the product of input positions `1:j`.

## Usage

``` r
prim_cumprod(x, axis)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of any data type.

- axis:

  (`integer(1)`)  
  Axis along which to accumulate. Negative values count from the end,
  i.e. `-1` refers to the last axis.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_reduce_window()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce_window.html)
with
[`hlo_multiply()`](https://r-xla.github.io/stablehlo/reference/hlo_multiply.html)
as the reducer.

## See also

[`nv_cumprod()`](https://r-xla.github.io/anvl/reference/nv_cumprod.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
prim_cumprod(x, axis = 1L)
#> AnvlArray
#>   1  3  5
#>   2 12 30
#> [ CPUi32{2,3} ] 
```
