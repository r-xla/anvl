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
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/dev/reference/common_dtype.md). Can
  be any numeric data type. Scalars are broadcast, and R values assume
  the other operand's data type within their [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  otherwise falling back to their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and being converted to the common data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape and common data type.

## See also

[`prim_pow()`](https://r-xla.github.io/anvl/dev/reference/prim_pow.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(2, 3, 4))
y <- nv_array(c(3, 2, 1))
x ^ y
#> AnvlArray
#>  8
#>  9
#>  4
#> [ CPUf32{3} ] 

# different data types are promoted to their common one
nv_pow(nv_scalar(2, "f32"), nv_scalar(3, "f64"))
#> AnvlArray
#>  8
#> [ CPUf64{} ] 

# a scalar is broadcast and an R integer is converted to a float
x^2L
#> AnvlArray
#>   4
#>   9
#>  16
#> [ CPUf32{3} ] 
```
