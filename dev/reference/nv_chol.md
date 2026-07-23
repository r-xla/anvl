# Cholesky Decomposition

Computes the Cholesky decomposition of a symmetric positive-definite
matrix. Supports batched inputs: axes before the last two are batch
axes.

## Usage

``` r
nv_chol(x, lower = FALSE)

# S3 method for class 'AnvlArray'
chol(x, ..., lower = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Symmetric positive-definite matrix with at least 2 axes. The last two
  axes form the square matrix; any leading axes are batch axes.

- lower:

  (`logical(1)`)  
  If `TRUE`, return the lower-triangular factor.

- ...:

  No additional arguments.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Triangular matrix with the same shape and data type as the input.

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md),
[`prim_chol()`](https://r-xla.github.io/anvl/dev/reference/prim_chol.md)

## Examples

``` r
a <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
nv_chol(a)
#> AnvlArray
#>  2.0000 1.0000
#>  0.0000 1.4142
#> [ CPUf32{2,2} ] 
```
