# Addition

Adds two arrays element-wise. You can also use the `+` operator.

## Usage

``` r
nv_add(lhs, rhs)
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

[`prim_add()`](https://r-xla.github.io/anvl/dev/reference/prim_add.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
nv_add(x, y)
#> AnvlArray
#>  5
#>  7
#>  9
#> [ CPUf32{3} ] 
x + y
#> AnvlArray
#>  5
#>  7
#>  9
#> [ CPUf32{3} ] 
```
