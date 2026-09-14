# Bitwise XOR

Element-wise bitwise XOR of two integer arrays, which for a boolean
array is the logical XOR. For *logical* inputs, you can also use `xor`.

## Usage

``` r
nv_xor(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and the promoted common data type of the inputs.

## See also

[`prim_xor()`](https://r-xla.github.io/anvl/dev/reference/prim_xor.md)
for the underlying primitive.

## Examples

``` r
nv_xor(nv_array(c(TRUE, FALSE, TRUE)), nv_array(c(TRUE, TRUE, FALSE)))
#> AnvlArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 
nv_xor(nv_array(12L), nv_array(10L)) # bitwise: 6
#> AnvlArray
#>  6
#> [ CPUi32{1} ] 
xor(nv_array(c(TRUE, FALSE)), nv_array(c(TRUE, TRUE))) # logical
#> AnvlArray
#>  0
#>  1
#> [ CPUbool{2} ] 
```
