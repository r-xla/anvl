# Bitwise OR

Element-wise bitwise OR – a logical OR on a boolean input, and a
bit-by-bit one on an integer. You can also use the `|` operator, which
expects `bool` inputs, however.

## Usage

``` r
nv_or(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/dev/reference/common_dtype.md). Can
  be any integerish data type. Scalars are broadcast, and R values
  assume the other operand's data type within their [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  otherwise falling back to their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and being converted to the common data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape and common data type.

## The `|` operator

`|` is *logical*, like in base R. Unlike base R it only accepts booleans
and does not auto-convert non-booleans by comparing them with 0.

## See also

[`prim_or()`](https://r-xla.github.io/anvl/dev/reference/prim_or.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
y <- nv_array(c(TRUE, TRUE, FALSE))
x | y
#> AnvlArray
#>  1
#>  1
#>  1
#> [ CPUbool{3} ] 

# different data types are promoted to their common one
nv_or(nv_scalar(12L, "i32"), nv_scalar(10L, "i64"))
#> AnvlArray
#>  14
#> [ CPUi64{} ] 

# a scalar is broadcast
x | TRUE
#> AnvlArray
#>  1
#>  1
#>  1
#> [ CPUbool{3} ] 
```
