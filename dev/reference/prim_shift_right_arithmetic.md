# Primitive Arithmetic Shift Right

Element-wise arithmetic right bit shift.

## Usage

``` r
prim_shift_right_arithmetic(x, shift)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array whose bits are shifted. Can be any integer data type.

- shift:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  By how many bits to shift each element of `x`. Has the same data type
  and shape as `x`. An R value takes the other operand's data type, and
  its [default data
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
[`hlo_shift_right_arithmetic()`](https://r-xla.github.io/stablehlo/reference/hlo_shift_right_arithmetic.html),
specified under
[shift_right_arithmetic](https://openxla.org/stablehlo/spec#shift_right_arithmetic).

## See also

[`nv_shift_right_arithmetic()`](https://r-xla.github.io/anvl/dev/reference/nv_shift_right_arithmetic.md)

## Examples

``` r
# two R values: both take an R integer's default data type
prim_shift_right_arithmetic(-32L, 2L)
#> AnvlArray
#>  -8
#> [ CPUi32{} ] 

# the R value is built at the array's data type instead
prim_shift_right_arithmetic(-32L, nv_scalar(2L, "i64"))
#> AnvlArray
#>  -8
#> [ CPUi64{} ] 
```
