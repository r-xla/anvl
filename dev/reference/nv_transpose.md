# Transpose

Permutes the dimensions of an array. You can also use
[`t()`](https://rdrr.io/r/base/t.html) for matrices.

## Usage

``` r
nv_transpose(x, permutation = NULL)

# S3 method for class 'AnvlArray'
t(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  New ordering of dimensions. If `NULL` (default), reverses the
  dimensions. Negative values count from the end, i.e. `-1` refers to
  the last dimension.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as `x` and shape `nv_shape(x)[permutation]`.

## See also

[`prim_transpose()`](https://r-xla.github.io/anvl/dev/reference/prim_transpose.md)
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
