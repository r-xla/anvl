# Transpose Cross Product (Matrix)

Computes `lhs %*% t(rhs)`. If `rhs` is missing, computes
`lhs %*% t(lhs)`.

## Usage

``` r
nv_tcrossprod(lhs, rhs = NULL)

# S3 method for class 'AnvlArray'
tcrossprod(x, y = NULL, ...)
```

## Arguments

- lhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  An array with at least 2 axes.

- rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) \|
  `NULL`)  
  Optional second array. If `NULL`, uses `lhs`.

- x, y:

  Same as `lhs` and `rhs`; the names used by the base R S3 generic.

- ...:

  No additional arguments.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)

## See also

[`nv_crossprod()`](https://r-xla.github.io/anvl/reference/nv_crossprod.md),
[`nv_matmul()`](https://r-xla.github.io/anvl/reference/nv_matmul.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2, dtype = "f32")
nv_tcrossprod(x)
#> AnvlArray
#>  35 44
#>  44 56
#> [ CPUf32{2,2} ] 
```
