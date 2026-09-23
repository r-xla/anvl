# Bitwise XOR

Element-wise bitwise XOR – a logical XOR on a boolean input, and a
bit-by-bit one on an integer. For *logical* inputs, you can also use
`xor`.

## Usage

``` r
nv_xor(x, y)
```

## Arguments

- x, y:

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

## See also

[`prim_xor()`](https://r-xla.github.io/anvl/dev/reference/prim_xor.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
y <- nv_array(c(TRUE, TRUE, FALSE))
nv_xor(x, y)
#> AnvlArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 

# different data types are promoted to their common one
nv_xor(nv_scalar(12L, "i32"), nv_scalar(10L, "i64"))
#> AnvlArray
#>  6
#> [ CPUi64{} ] 

# a scalar is broadcast
nv_xor(x, TRUE)
#> AnvlArray
#>  0
#>  1
#>  0
#> [ CPUbool{3} ] 
```
