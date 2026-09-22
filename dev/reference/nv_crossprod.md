# Cross Product (Matrix)

Computes `t(lhs) %*% rhs`. If `rhs` is missing, computes
`t(lhs) %*% lhs`. Above rank 2 the last two axes are the matrix and the
leading ones are batch axes, as in
[`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md):
only the matrix is transposed.

## Usage

``` r
nv_crossprod(lhs, rhs = NULL)

# S3 method for class 'AnvlArray'
crossprod(x, y = NULL, ...)
```

## Arguments

- lhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array with at least 2 axes, as for
  [`base::crossprod()`](https://rdrr.io/r/base/crossprod.html). Can be
  any numeric data type; `lhs` and `rhs` are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).

- rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \| `NULL`)  
  Optional second array. If `NULL`, uses `lhs`.

- x, y:

  Same as `lhs` and `rhs`; the names used by the base R S3 generic.

- ...:

  No additional arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the operands' common data type, and the shape of `t(lhs) %*% rhs`.

## See also

[`nv_tcrossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_tcrossprod.md),
[`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md)

## Examples

``` r
# `t(x) %*% x`, so a 3x2 gives a 2x2
x <- nv_matrix(1:6, nrow = 3, dtype = "f32")
nv_crossprod(x)
#> AnvlArray
#>  14 32
#>  32 77
#> [ CPUf32{2,2} ] 
```
