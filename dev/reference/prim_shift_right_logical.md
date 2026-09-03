# Primitive Logical Shift Right

Element-wise logical right bit shift.

## Usage

``` r
prim_shift_right_logical(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of data type boolean, integer, or unsigned integer.
  Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_shift_right_logical()`](https://r-xla.github.io/stablehlo/reference/hlo_shift_right_logical.html).

## See also

[`nv_shift_right_logical()`](https://r-xla.github.io/anvl/dev/reference/nv_shift_right_logical.md)

## Examples

``` r
x <- nv_array(c(8L, 16L, 32L))
y <- nv_array(c(1L, 2L, 3L))
prim_shift_right_logical(x, y)
#> AnvlArray
#>  4
#>  4
#>  4
#> [ CPUi32{3} ] 
```
