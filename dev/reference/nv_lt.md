# Less Than

Element-wise less than comparison. You can also use the `<` operator.

## Usage

``` r
nv_lt(lhs, rhs)
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

[`prim_lt()`](https://r-xla.github.io/anvl/dev/reference/prim_lt.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(3, 2, 1))
nv_lt(x, y)
#> AnvlArray
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 
x < y
#> AnvlArray
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 
```
