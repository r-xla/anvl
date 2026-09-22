# Less Than or Equal

Element-wise less than or equal comparison. You can also use the `<=`
operator.

## Usage

``` r
nv_le(lhs, rhs)
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
Has the inputs' broadcast shape and boolean data type.

## See also

[`prim_le()`](https://r-xla.github.io/anvl/dev/reference/prim_le.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(3, 2, 1))
nv_le(x, y)
#> AnvlArray
#>  1
#>  1
#>  0
#> [ CPUbool{3} ] 
x <= y
#> AnvlArray
#>  1
#>  1
#>  0
#> [ CPUbool{3} ] 

# different data types are promoted to their common one
nv_le(nv_scalar(1, "f32"), nv_scalar(2, "f64"))
#> AnvlArray
#>  1
#> [ CPUbool{} ] 

# a scalar is broadcast and an R integer is converted to a float
x <= 2L
#> AnvlArray
#>  1
#>  1
#>  0
#> [ CPUbool{3} ] 
```
