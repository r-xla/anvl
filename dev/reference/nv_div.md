# Division

Divides two arrays element-wise. You can also use the `/` operator.

## Usage

``` r
nv_div(lhs, rhs)
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

[`prim_div()`](https://r-xla.github.io/anvl/dev/reference/prim_div.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(10, 20, 30))
y <- nv_array(c(2, 5, 10))
nv_div(x, y)
#> AnvlArray
#>  5
#>  4
#>  3
#> [ CPUf32{3} ] 
x / y
#> AnvlArray
#>  5
#>  4
#>  3
#> [ CPUf32{3} ] 
```
