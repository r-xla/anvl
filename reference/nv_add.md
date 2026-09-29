# Addition

Adds two arrays element-wise. You can also use the `+` operator.

## Usage

``` r
nv_add(lhs, rhs)

# S3 method for class 'AnvlArray'
e1 + e2
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
Has the inputs' broadcast shape and common data type.

## See also

[`prim_add()`](https://r-xla.github.io/anvl/reference/prim_add.md) for
the underlying primitive.

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
