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
  One input, a symmetric positive-definite matrix with at least 2 axes,
  the last two forming the square matrix and any leading ones batch
  axes. Can be any numeric data type: a float keeps its own, and an
  integer one is converted to the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is converted in the same way.

- lower:

  (`logical(1)`)  
  If `TRUE`, return the lower-triangular factor.

- ...:

  No additional arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Triangular matrix with the input's shape, and its data type – or the
default float data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where the input was an integer one. The values in the triangle not
selected by `lower` are implementation-defined.

## Details

Differentiation is only implemented for a single matrix: a
[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
of a batched decomposition errors.

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md),
[`prim_chol()`](https://r-xla.github.io/anvl/dev/reference/prim_chol.md)

## Examples

``` r
# the factor has the matrix's shape and data type
a <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
nv_chol(a)
#> AnvlArray
#>  2.0000 1.0000
#>  0.0000 1.4142
#> [ CPUf32{2,2} ] 

# an integer matrix is factored at the default float data type
nv_chol(nv_matrix(c(4L, 2L, 2L, 3L), nrow = 2))
#> AnvlArray
#>  2.0000 1.0000
#>  0.0000 1.4142
#> [ CPUf32{2,2} ] 
```
