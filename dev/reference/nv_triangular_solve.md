# Triangular Solve

Solves a triangular system of linear equations. When `left = TRUE`,
returns `x` such that `op(a) %*% x = b`. When `left = FALSE`, returns
`x` such that `x %*% op(a) = b`. Here `op` is `a` or `t(a)` depending on
`transpose`.

## Usage

``` r
nv_triangular_solve(
  a,
  b,
  left = TRUE,
  lower = TRUE,
  unit_diag = FALSE,
  transpose = FALSE
)
```

## Arguments

- a:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Triangular coefficient matrix with at least 2 axes. The first two axes
  must be of equal size; any trailing axes are batch axes. Can be any
  numeric data type: `a` and `b` are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md)
  and that is then converted to the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
  where it is not a float already, since the solve is a float one. An R
  value assumes the other operand's data type within its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtype_categories.md),
  and settles on the default float when neither has one.

- b:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Right-hand side. For `a` of shape `(n, n, b1, ..., bk)`, `b` may be
  either:

  - full rank — shape `(n, m, b1, ..., bk)` when `left = TRUE`, or
    `(m, n, b1, ..., bk)` when `left = FALSE`;

  - one rank less, shape `(n, b1, ..., bk)`, meaning a single column
    (`left = TRUE`) or row (`left = FALSE`) per batch — it is reshaped
    internally and the reshape is undone on the result so the output
    rank matches `b`.

  `b`'s trailing batch axes (`b1, ..., bk`) must match `a`'s exactly. It
  is promoted together with `a` – see `a`.

- left:

  (`logical(1)`)  
  If `TRUE` (default), solve `op(a) %*% x = b`; if `FALSE`, solve
  `x %*% op(a) = b`.

- lower:

  (`logical(1)`)  
  Whether `a` is lower or upper triangular. Defaults to `TRUE`.

- unit_diag:

  (`logical(1)`)  
  If `TRUE`, the diagonal of `a` is treated as all ones (and the actual
  values on the diagonal are ignored). Defaults to `FALSE`.

- transpose:

  (`logical(1)`)  
  If `TRUE`, solve with `t(a)` in place of `a`. Defaults to `FALSE`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
The solution `x`, with `b`'s shape and the operands' common data type –
or the default float data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where that was an integer one.

## Details

As a convenience, `b` may have one fewer axis than `a` (a single
right-hand side per batch, shape `(n, b1, ..., bk)` for `a` of shape
`(n, n, b1, ..., bk)`). It is reshaped internally to a column
(`left = TRUE`) or row (`left = FALSE`) and reshaped back on the way
out. Because we don't broadcast, this is not ambiguous (as it would be
for NumPy).

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
