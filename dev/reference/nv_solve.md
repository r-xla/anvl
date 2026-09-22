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

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Square non-singular matrix with exactly 2 axes. Can be any numeric
  data type: `a` and `b` are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md)
  and that is then converted to the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
  where it is not a float already, since the decomposition is a float
  one. An R value assumes the other operand's data type within its [data
  type category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  and settles on the default float when neither has one.

- b:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Right-hand side, vector of length `n` or matrix with `n` rows.
  Promoted together with `a` – see `a`.

- ...:

  No additional arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
The solution `x` such that `a %*% x = b`, with `b`'s shape and the
operands' common data type – or the default float data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where that was an integer one.

## Details

\$\$A x = b\$\$ \$\$P A = L U\$\$ \$\$L U x = P b\$\$ \$\$L y = P b\$\$
\$\$U x = y\$\$

## Shapes

- `a`: `(n, n)`

- `b`: `(n,)` or `(n, k)`

- output: same shape as `b`

## See also

[`nv_chol()`](https://r-xla.github.io/anvl/dev/reference/nv_chol.md),
[`nv_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_triangular_solve.md),
[`prim_lu()`](https://r-xla.github.io/anvl/dev/reference/prim_lu.md)

## Examples

``` r
# the solution has `b`'s shape and the operands' common data type
a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
b <- nv_matrix(c(1, 2), nrow = 2, dtype = "f64")
nv_solve(a, b)
#> AnvlArray
#>   1.5000
#>  -0.8333
#> [ CPUf64{2,1} ] 

# an integer system is solved at the default float data type
nv_solve(nv_matrix(c(3L, 1L, 1L, 2L), nrow = 2), nv_array(c(9L, 8L)))
#> AnvlArray
#>  2.0000
#>  3.0000
#> [ CPUf32{2} ] 
```
