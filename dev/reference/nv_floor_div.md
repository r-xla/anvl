# Flooring Division

Element-wise flooring division. You can also call this via the `%/%`
operator. The result is the largest whole number that does not exceed
`lhs / rhs`.

## Usage

``` r
nv_floor_div(lhs, rhs)
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

[`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md) for
the matching remainder,
[`nv_div()`](https://r-xla.github.io/anvl/dev/reference/nv_div.md) for
the division itself.

## Examples

``` r
x <- nv_array(c(7L, -7L))
y <- nv_array(c(2L, 2L))
nv_floor_div(x, y)
#> AnvlArray
#>   3
#>  -4
#> [ CPUi32{2} ] 
x %/% y
#> AnvlArray
#>   3
#>  -4
#> [ CPUi32{2} ] 
```
