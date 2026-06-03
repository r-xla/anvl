# Modulo (Flooring Remainder)

Element-wise flooring remainder of division. The sign of the result
equals the sign of `rhs`, matching base R's `%%` operator.

## Usage

``` r
nv_mod(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvl/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvl/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and the promoted common data type of the inputs.

## See also

[`nv_remainder()`](https://r-xla.github.io/anvl/reference/nv_remainder.md)
for truncating remainder,
[`prim_remainder()`](https://r-xla.github.io/anvl/reference/prim_remainder.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1L, -1L))
y <- nv_array(c(-3L, 3L))
nv_mod(x, y)
#> AnvlArray
#>  -2
#>   2
#> [ CPUi32{2} ] 
as.vector(x) %% as.vector(y)
#> [1] -2  2
```
