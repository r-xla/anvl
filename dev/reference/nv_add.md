# Addition

Adds two arrays element-wise. You can also use the `+` operator.

## Usage

``` r
nv_add(lhs, rhs)
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

# different data types are promoted to their common one
nv_add(nv_scalar(1, "f32"), nv_scalar(2, "f64"))
#> AnvlArray
#>  3
#> [ CPUf64{} ] 

# a scalar is broadcast and an R integer is converted to a float
x + 1L
#> AnvlArray
#>  2
#>  3
#>  4
#> [ CPUf32{3} ] 
```
