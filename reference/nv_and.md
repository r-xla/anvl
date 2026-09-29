# Bitwise AND

Element-wise bitwise AND – a logical AND on a boolean input, and a
bit-by-bit one on an integer. You can also use the `&` operator, which
expects `bool` inputs, however.

## Usage

``` r
nv_and(lhs, rhs)

# S3 method for class 'AnvlArray'
e1 & e2
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/reference/common_dtype.md). Can be
  any integerish data type. Scalars are broadcast. An R value takes the
  other operand's data type when that is in its own or a higher
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

## The `&` operator

`&` is *logical*, like in base R. Unlike base R it only accepts booleans
and does not auto-convert non-booleans by comparing them with 0.

## See also

[`prim_and()`](https://r-xla.github.io/anvl/reference/prim_and.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
y <- nv_array(c(TRUE, TRUE, FALSE))
x & y
#> AnvlArray
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 

# different data types are promoted to their common one
nv_and(nv_scalar(12L, "i32"), nv_scalar(10L, "i64"))
#> AnvlArray
#>  8
#> [ CPUi64{} ] 

# a scalar is broadcast
x & TRUE
#> AnvlArray
#>  1
#>  0
#>  1
#> [ CPUbool{3} ] 
```
