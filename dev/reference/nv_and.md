# Bitwise AND

Element-wise bitwise AND of two integer arrays, which for a boolean
array is the logical AND.

## Usage

``` r
nv_and(lhs, rhs)
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

## The `&` operator

`&` is *logical*, like in base R. Unlike base R it only accepts booleans
and does not auto-convert non-booleans by comparing them with 0.

## See also

[`prim_and()`](https://r-xla.github.io/anvl/dev/reference/prim_and.md)
for the underlying primitive.

## Examples

``` r
nv_and(nv_array(c(TRUE, FALSE, TRUE)), nv_array(c(TRUE, TRUE, FALSE)))
#> AnvlArray
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 
nv_and(nv_array(12L), nv_array(10L)) # bitwise: 8
#> AnvlArray
#>  8
#> [ CPUi32{1} ] 
nv_array(c(TRUE, FALSE)) & nv_array(c(TRUE, TRUE)) # logical
#> AnvlArray
#>  1
#>  0
#> [ CPUbool{2} ] 
```
