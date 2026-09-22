# Arithmetic Shift Right

Element-wise arithmetic right bit shift.

## Usage

``` r
nv_shift_right_arithmetic(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/dev/reference/common_dtype.md). Can
  be any integer data type. Scalars are broadcast, and R values assume
  the other operand's data type within their [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  otherwise falling back to their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and being converted to the common data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape and common data type.

## See also

[`prim_shift_right_arithmetic()`](https://r-xla.github.io/anvl/dev/reference/prim_shift_right_arithmetic.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(8L, -16L, 32L))
y <- nv_array(c(1L, 2L, 3L))
nv_shift_right_arithmetic(x, y)
#> AnvlArray
#>   4
#>  -4
#>   4
#> [ CPUi32{3} ] 

# different data types are promoted to their common one
nv_shift_right_arithmetic(nv_scalar(-32L, "i32"), nv_scalar(2L, "i64"))
#> AnvlArray
#>  -8
#> [ CPUi64{} ] 

# a scalar is broadcast
nv_shift_right_arithmetic(x, 1L)
#> AnvlArray
#>   4
#>  -8
#>  16
#> [ CPUi32{3} ] 
```
