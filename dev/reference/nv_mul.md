# Multiplication

Multiplies two arrays element-wise. You can also use the `*` operator.

## Usage

``` r
nv_mul(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/dev/reference/common_dtype.md). Can
  be any data type. Scalars are broadcast, and R values assume the other
  operand's data type within their [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  otherwise falling back to their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and being converted to the common data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape and common data type.

## See also

[`prim_mul()`](https://r-xla.github.io/anvl/dev/reference/prim_mul.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
nv_mul(x, y)
#> AnvlArray
#>   4
#>  10
#>  18
#> [ CPUf32{3} ] 
x * y
#> AnvlArray
#>   4
#>  10
#>  18
#> [ CPUf32{3} ] 

# different data types are promoted to their common one
nv_mul(nv_scalar(2, "f32"), nv_scalar(3, "f64"))
#> AnvlArray
#>  6
#> [ CPUf64{} ] 

# a scalar is broadcast and an R integer is converted to a float
x * 2L
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUf32{3} ] 
```
