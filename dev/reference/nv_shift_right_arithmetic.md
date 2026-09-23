# Arithmetic Shift Right

Element-wise arithmetic right bit shift.

## Usage

``` r
nv_shift_right_arithmetic(x, shift)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array whose bits are shifted. Can be any integer data type.

- shift:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  By how many bits to shift each element of `x`. Brought to `x`'s data
  type, which it must fit in. Scalars are broadcast.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape and `x`'s data type.

## See also

[`prim_shift_right_arithmetic()`](https://r-xla.github.io/anvl/dev/reference/prim_shift_right_arithmetic.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(8L, -16L, 32L))
shift <- nv_array(c(1L, 2L, 3L))
nv_shift_right_arithmetic(x, shift)
#> AnvlArray
#>   4
#>  -4
#>   4
#> [ CPUi32{3} ] 

# the result keeps `x`'s data type, which `shift` is brought to
nv_shift_right_arithmetic(nv_scalar(-32L, "i64"), nv_scalar(2L, "i32"))
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
