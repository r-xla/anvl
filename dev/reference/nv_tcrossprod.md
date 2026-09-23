# Transpose Cross Product (Matrix)

Computes `x %*% t(y)`. If `y` is missing, computes `x %*% t(x)`. Above
rank 2 the last two axes are the matrix and the leading ones are batch
axes, as in
[`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md):
only the matrix is transposed.

## Usage

``` r
nv_tcrossprod(x, y = NULL)

# S3 method for class 'AnvlArray'
tcrossprod(x, y = NULL, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array with at least 2 axes, as for
  [`base::tcrossprod()`](https://rdrr.io/r/base/crossprod.html). Can be
  any numeric data type; `x` and `y` are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).

- y:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \| `NULL`)  
  Optional second array. If `NULL`, uses `x`.

- ...:

  No additional arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the operands' common data type, and the shape of `x %*% t(y)`.

## See also

[`nv_crossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_crossprod.md),
[`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md)

## Examples

``` r
# `x %*% t(x)`, so a 2x3 gives a 2x2
x <- nv_matrix(1:6, nrow = 2, dtype = "f32")
nv_tcrossprod(x)
#> AnvlArray
#>  35 44
#>  44 56
#> [ CPUf32{2,2} ] 
```
