# Greater Than

Element-wise greater than comparison. You can also use the `>` operator.

## Usage

``` r
nv_gt(lhs, rhs)
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
Has the same shape as the inputs and boolean data type.

## See also

[`prim_gt()`](https://r-xla.github.io/anvl/dev/reference/prim_gt.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(3, 2, 1))
nv_gt(x, y)
#> AnvlArray
#>  0
#>  0
#>  1
#> [ CPUbool{3} ] 
x > y
#> AnvlArray
#>  0
#>  0
#>  1
#> [ CPUbool{3} ] 
```
