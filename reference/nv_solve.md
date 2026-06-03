# Solve Linear System

Solves the linear system `a %*% x = b` for `x`. Uses LU decomposition
with partial pivoting internally, so `a` need only be square and
non-singular.

## Usage

``` r
nv_solve(a, b)

# S3 method for class 'AnvlArray'
solve(a, b, ...)
```

## Arguments

- a:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Coefficient matrix.

- b:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Right-hand side. If missing, returns
  [`nv_inv()`](https://r-xla.github.io/anvl/reference/nv_inv.md) of `a`.

- ...:

  No additional arguments.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
The solution `x` such that `a %*% x = b`.

## Details

\$\$A x = b\$\$ \$\$P A = L U\$\$ \$\$L U x = P b\$\$ \$\$L y = P b\$\$
\$\$U x = y\$\$

## Shapes

- `a`: `(n, n)`

- `b`: `(n,)` or `(n, k)`

- output: same shape as `b`

## See also

[`nv_chol()`](https://r-xla.github.io/anvl/reference/nv_chol.md),
[`nv_triangular_solve()`](https://r-xla.github.io/anvl/reference/nv_triangular_solve.md),
[`prim_lu()`](https://r-xla.github.io/anvl/reference/prim_lu.md)

## Examples

``` r
a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
b <- nv_matrix(c(1, 2), nrow = 2, dtype = "f64")
nv_solve(a, b)
#> AnvlArray
#>   1.5000
#>  -0.8333
#> [ CPUf64{2,1} ] 
```
