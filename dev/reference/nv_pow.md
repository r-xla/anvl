# Power

Raises `lhs` to the power of `rhs` element-wise. You can also use the
`^` operator.

## Usage

``` r
nv_pow(lhs, rhs)
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

[`prim_pow()`](https://r-xla.github.io/anvl/dev/reference/prim_pow.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(2, 3, 4))
y <- nv_array(c(3, 2, 1))
nv_pow(x, y)
#> AnvlArray
#>  8
#>  9
#>  4
#> [ CPUf32{3} ] 
x^y
#> AnvlArray
#>  8
#>  9
#>  4
#> [ CPUf32{3} ] 
```
