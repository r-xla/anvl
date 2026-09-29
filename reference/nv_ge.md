# Greater Than or Equal

Element-wise greater than or equal comparison. You can also use the `>=`
operator.

## Usage

``` r
nv_ge(lhs, rhs)

# S3 method for class 'AnvlArray'
e1 >= e2
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/reference/common_dtype.md). Can be
  any data type. Scalars are broadcast. An R value takes the other
  operand's data type when that is in its own or a higher
  [category](https://r-xla.github.io/anvl/reference/dtype_categories.md).
  Otherwise it settles on its [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md), and
  the operands meet at their common data type.

- e1, e2:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  The operands of the operator, which it passes on as `lhs` and `rhs`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
Has the inputs' broadcast shape and boolean data type.

## See also

[`prim_ge()`](https://r-xla.github.io/anvl/reference/prim_ge.md) for the
underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(3, 2, 1))
nv_ge(x, y)
#> AnvlArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 
x >= y
#> AnvlArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 

# different data types are promoted to their common one
nv_ge(nv_scalar(2, "f32"), nv_scalar(1, "f64"))
#> AnvlArray
#>  1
#> [ CPUbool{} ] 

# a scalar is broadcast and an R integer is converted to a float
x >= 2L
#> AnvlArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 
```
