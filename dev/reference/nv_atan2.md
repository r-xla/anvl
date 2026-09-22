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
  Two inputs. Can be any numeric data type: the two are first brought to
  a [common data
  type](https://r-xla.github.io/anvl/dev/reference/common_dtype.md) and
  that is then converted to the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
  where it is not a float already, so the result is always a float.
  Scalars are broadcast. An R value assumes the other operand's data
  type within its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  settles on the default float when neither operand has one.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape, and their common data type – or the
default float data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where that was an integer one.

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

# different data types are promoted to their common one
nv_atan2(nv_scalar(1, "f32"), nv_scalar(1, "f64"))
#> AnvlArray
#>  0.7854
#> [ CPUf64{} ] 

# a scalar is broadcast and an R integer is converted to a float
nv_atan2(y, 1L)
#> AnvlArray
#>   0.7854
#>   0.0000
#>  -0.7854
#> [ CPUf32{3} ] 
```
