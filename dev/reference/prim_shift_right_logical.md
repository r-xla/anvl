# Primitive Logical Shift Right

Element-wise logical right bit shift.

## Usage

``` r
prim_shift_right_logical(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any integer data
  type. R values assume the other operand's data type when it is in
  their [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when neither operand has one.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' shape and data type.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_shift_right_logical()`](https://r-xla.github.io/stablehlo/reference/hlo_shift_right_logical.html),
specified under
[shift_right_logical](https://openxla.org/stablehlo/spec#shift_right_logical).

## See also

[`nv_shift_right_logical()`](https://r-xla.github.io/anvl/dev/reference/nv_shift_right_logical.md)

## Examples

``` r
# two R values: both take an R integer's default data type
prim_shift_right_logical(32L, 2L)
#> AnvlArray
#>  8
#> [ CPUi32{} ] 

# the R value is built at the array's data type instead
prim_shift_right_logical(32L, nv_scalar(2L, "i64"))
#> AnvlArray
#>  8
#> [ CPUi64{} ] 
```
