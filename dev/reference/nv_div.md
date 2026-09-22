# Division

Divides two arrays element-wise. You can also use the `/` operator.

## Usage

``` r
nv_div(lhs, rhs)
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

# different data types are promoted to their common one
nv_div(nv_scalar(10, "f32"), nv_scalar(4, "f64"))
#> AnvlArray
#>  2.5000
#> [ CPUf64{} ] 

# a scalar is broadcast and an R integer is converted to a float
x / 2L
#> AnvlArray
#>   5
#>  10
#>  15
#> [ CPUf32{3} ] 
```
