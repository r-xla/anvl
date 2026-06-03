# Transpose

Permutes the dimensions of an array. You can also use
[`t()`](https://rdrr.io/r/base/t.html) for matrices.

## Usage

``` r
nv_transpose(operand, permutation = NULL)

# S3 method for class 'AnvlArray'
t(x)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  New ordering of dimensions. If `NULL` (default), reverses the
  dimensions.

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Same as `operand`; this is the name used by the base R S3 generic.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same data type as `operand` and shape
`nv_shape(operand)[permutation]`.

## See also

[`prim_transpose()`](https://r-xla.github.io/anvl/reference/prim_transpose.md)
for the underlying primitive.

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
t(x)
#> AnvlArray
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
