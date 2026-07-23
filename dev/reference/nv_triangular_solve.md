# Triangular Solve

Solves a triangular system of linear equations. When `left_side = TRUE`,
returns `x` such that `op(a) %*% x = b`. When `left_side = FALSE`,
returns `x` such that `x %*% op(a) = b`. Here `op` is `a` or `t(a)`
depending on `transpose_a`.

## Usage

``` r
nv_triangular_solve(
  a,
  b,
  left_side = TRUE,
  lower = TRUE,
  unit_diagonal = FALSE,
  transpose_a = FALSE
)
```

## Arguments

- a:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Triangular coefficient matrix with at least 2 axes. The last two axes
  must be equal; any leading axes are batch axes.

- b:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Right-hand side. For `a` of shape `(B..., n, n)`, `b` may be either:

  - full rank — shape `(B..., n, k)` when `left_side = TRUE`, or
    `(B..., k, n)` when `left_side = FALSE`;

  - one rank less, shape `(B..., n)`, meaning a single column
    (`left_side = TRUE`) or row (`left_side = FALSE`) per batch — it is
    reshaped internally and the reshape is undone on the result so the
    output rank matches `b`.

  `b`'s batch axes (`B...`) must match `a`'s exactly.

- left_side:

  (`logical(1)`)  
  If `TRUE` (default), solve `op(a) %*% x = b`; if `FALSE`, solve
  `x %*% op(a) = b`.

- lower:

  (`logical(1)`)  
  Whether `a` is lower or upper triangular. Defaults to `TRUE`.

- unit_diagonal:

  (`logical(1)`)  
  If `TRUE`, the diagonal of `a` is treated as all ones (and the actual
  values on the diagonal are ignored). Defaults to `FALSE`.

- transpose_a:

  (`logical(1)`)  
  If `TRUE`, solve with `t(a)` in place of `a`. Defaults to `FALSE`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
The solution `x`, with the same shape and dtype as `b`.

## Details

As a convenience, `b` may have one fewer axis than `a` (a single
right-hand side per batch, shape `(B..., n)` for `a` of shape
`(B..., n, n)`). It is reshaped internally to a column
(`left_side = TRUE`) or row (`left_side = FALSE`) and reshaped back on
the way out. Because we don't broadcast, this is not ambiguous (as it
would be for NumPy).

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md),
[`nv_chol()`](https://r-xla.github.io/anvl/dev/reference/nv_chol.md),
[`prim_triangular_solve()`](https://r-xla.github.io/anvl/dev/reference/prim_triangular_solve.md)

## Examples

``` r
L <- nv_matrix(c(2, 1, 0, 3), nrow = 2, dtype = "f32")
b <- nv_matrix(c(4, 3), nrow = 2, dtype = "f32")
nv_triangular_solve(L, b)
#> AnvlArray
#>  2.0000
#>  0.3333
#> [ CPUf32{2,1} ] 
```
