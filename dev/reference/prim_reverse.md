# Primitive Reverse

Reverses the order of elements along specified axes.

## Usage

``` r
prim_reverse(x, axes)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes to reverse. Negative values count from the end, i.e. `-1` refers
  to the last axis.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type and shape as `x`. It is ambiguous if the input is
ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_reverse()`](https://r-xla.github.io/stablehlo/reference/hlo_reverse.html).

## See also

[`nv_reverse()`](https://r-xla.github.io/anvl/dev/reference/nv_reverse.md)

## Examples

``` r
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
