# Primitive Reverse

Reverses the order of elements along specified axes.

## Usage

``` r
prim_reverse(x, axes)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes to reverse. Negative values count from the end, i.e. `-1` refers
  to the last axis.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type and shape.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_reverse()`](https://r-xla.github.io/stablehlo/reference/hlo_reverse.html),
specified under [reverse](https://openxla.org/stablehlo/spec#reverse).

## See also

[`nv_reverse()`](https://r-xla.github.io/anvl/dev/reference/nv_reverse.md)

## Examples

``` r
# the order along axis 1 is flipped
x <- nv_array(c(1, 2, 3, 4, 5))
prim_reverse(x, axes = 1L)
#> AnvlArray
#>  5
#>  4
#>  3
#>  2
#>  1
#> [ CPUf32{5} ] 
```
