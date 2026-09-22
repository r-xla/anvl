# Maximum

Element-wise maximum of two arrays.

## Usage

``` r
nv_max(lhs, rhs)
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

[`prim_max()`](https://r-xla.github.io/anvl/dev/reference/prim_max.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 5, 3))
y <- nv_array(c(4, 2, 6))
nv_max(x, y)
#> AnvlArray
#>  4
#>  5
#>  6
#> [ CPUf32{3} ] 

# different data types are promoted to their common one
nv_max(nv_scalar(1, "f32"), nv_scalar(5, "f64"))
#> AnvlArray
#>  5
#> [ CPUf64{} ] 

# a scalar is broadcast and an R integer is converted to a float
nv_max(x, 2L)
#> AnvlArray
#>  2
#>  5
#>  3
#> [ CPUf32{3} ] 
```
