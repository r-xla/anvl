# Primitive Cholesky Decomposition

Computes the Cholesky decomposition of a symmetric positive-definite
matrix. Dimensions before the last two are batch dimensions.

## Usage

``` r
prim_chol(operand, lower = FALSE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of data type floating-point with at least 2 dimensions.
  The last two dimensions must be equal (square matrix); any leading
  dimensions are batch dimensions.

- lower:

  (`logical(1)`)  
  If `FALSE` (default, matching base R's
  [`base::chol()`](https://rdrr.io/r/base/chol.html)), compute the upper
  triangular factor `U` such that `operand = t(U) %*% U`. If `TRUE`,
  compute the lower triangular factor `L` such that
  `operand = L %*% t(L)`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input. The values in the
triangle not specified by `lower` are implementation-defined. It is
ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_cholesky()`](https://r-xla.github.io/stablehlo/reference/hlo_cholesky.html).

## References

Murray, Iain (2016). “Differentiation of the Cholesky decomposition.”
*arXiv preprint arXiv:1602.07527*.

Walter, Sebastian (2012). *Structured higher-order algorithmic
differentiation in the forward and reverse mode with application in
optimum experimental design*. Ph.D. thesis,
Mathematisch-Naturwissenschaftliche Fakult"at II.

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/reference/nv_solve.md)

## Examples

``` r
# Create a positive-definite matrix
x <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
prim_chol(x, lower = TRUE)
#> AnvlArray
#>  2.0000 0.0000
#>  1.0000 1.4142
#> [ CPUf32{2,2} ] 
```
