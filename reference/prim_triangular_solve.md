# Primitive Triangular Solve

Solves a system of linear equations with a triangular coefficient
matrix. When `left_side` is `TRUE`, solves `op(a) %*% x = b` for `x`.
When `left_side` is `FALSE`, solves `x %*% op(a) = b` for `x`.
Dimensions before the last two are batch dimensions and must match
between `a` and `b` (no broadcasting). Here `op` is `A` or `A^T`
depending on `transpose_a`.

## Usage

``` r
prim_triangular_solve(a, b, left_side, lower, unit_diagonal, transpose_a)
```

## Arguments

- a:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Triangular coefficient matrix of data type floating-point with at
  least 2 dimensions. The last two dimensions must be equal (square
  matrix); any leading dimensions are batch dimensions.

- b:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Right-hand side. Same data type and rank as `a` (rank \>= 2), with
  matching leading batch dimensions. The size of `a`'s last two (square)
  dimensions must equal `b`'s second-to-last dimension when
  `left_side = TRUE`, or `b`'s last dimension when `left_side = FALSE`.

- left_side:

  (`logical(1)`)  
  If `TRUE`, solve `op(a) %*% x = b`. If `FALSE`, solve
  `x %*% op(a) = b`.

- lower:

  (`logical(1)`)  
  If `TRUE`, `a` is lower triangular. If `FALSE`, `a` is upper
  triangular.

- unit_diagonal:

  (`logical(1)`)  
  If `TRUE`, assume diagonal elements of `a` are 1.

- transpose_a:

  (`logical(1)`)  
  If `TRUE`, solve with `t(a)` in place of `a`. Defaults to `FALSE`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as `b`. It is ambiguous if both `a` and
`b` are ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_triangular_solve()`](https://r-xla.github.io/stablehlo/reference/hlo_triangular_solve.html).

## References

Giles, Mike (2008). “An extended collection of matrix derivative results
for forward and reverse mode automatic differentiation.” Oxford
University Computing Laboratory.

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/reference/nv_solve.md)

## Examples

``` r
# Solve L %*% x = b where L is lower triangular
L <- nv_matrix(c(2, 0, 1, 3), nrow = 2, dtype = "f32")
b <- nv_matrix(c(4, 3), nrow = 2, dtype = "f32")
prim_triangular_solve(L, b,
  left_side = TRUE, lower = TRUE,
  unit_diagonal = FALSE, transpose_a = FALSE
)
#> AnvlArray
#>  2
#>  1
#> [ CPUf32{2,1} ] 
```
