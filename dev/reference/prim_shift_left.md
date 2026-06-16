# Primitive Shift Left

Element-wise left bit shift.

## Usage

``` r
prim_shift_left(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of data type boolean, integer, or unsigned integer.
  Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_shift_left()`](https://r-xla.github.io/stablehlo/reference/hlo_shift_left.html).

## See also

[`nv_shift_left()`](https://r-xla.github.io/anvl/dev/reference/nv_shift_left.md)

## Examples

``` r
x <- nv_array(c(1L, 2L, 4L))
y <- nv_array(c(1L, 2L, 1L))
prim_shift_left(x, y)
#> AnvlArray
#>  2
#>  8
#>  8
#> [ CPUi32{3} ] 
```
