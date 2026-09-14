# Arctangent 2

Element-wise two-argument arctangent, i.e. the angle (in radians)
between the positive x-axis and the point `(rhs, lhs)`.

## Usage

``` r
nv_atan2(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Left and right operand. An integer operand is converted to the default
  float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  Operands are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and the promoted common data type of the inputs.

## See also

[`prim_atan2()`](https://r-xla.github.io/anvl/dev/reference/prim_atan2.md)
for the underlying primitive.

## Examples

``` r
y <- nv_array(c(1, 0, -1))
x <- nv_array(c(0, 1, 0))
nv_atan2(y, x)
#> AnvlArray
#>   1.5708
#>   0.0000
#>  -1.5708
#> [ CPUf32{3} ] 
```
