# Primitive Cholesky Decomposition

Computes the Cholesky decomposition of a symmetric positive-definite
matrix. The first two axes form the matrix, and any trailing axes are
batch axes.

## Usage

``` r
prim_chol(x, lower = FALSE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with at least 2 axes, the first two of equal size (a square
  matrix); any trailing axes are batch axes. Can be any float data type.
  An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- lower:

  (`logical(1)`)  
  If `FALSE` (default, matching base R's
  [`base::chol()`](https://rdrr.io/r/base/chol.html)), compute the upper
  triangular factor `U` such that `x = t(U) %*% U`. If `TRUE`, compute
  the lower triangular factor `L` such that `x = L %*% t(L)`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the same shape and data type as the input. The values in the
triangle not specified by `lower` are implementation-defined. A matrix
that is not positive definite factors to `NaN`.

## Implemented Rules

- `stablehlo`

- `pullback`

- `vectorize`

## StableHLO

Lowers to a `"potrf"`
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
(backed by LAPACK on CPU and cuSOLVER on CUDA) + postprocessing.

## References

Murray I (2016). “Differentiation of the Cholesky decomposition.” *arXiv
preprint arXiv:1602.07527*.

Walter S (2012). *Structured higher-order algorithmic differentiation in
the forward and reverse mode with application in optimum experimental
design*. Ph.D. thesis, Mathematisch-Naturwissenschaftliche Fakult"at II.

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)

## Examples

``` r
# create a positive-definite matrix
x <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
prim_chol(x, lower = TRUE)
#> AnvlArray
#>  2.0000 0.0000
#>  1.0000 1.4142
#> [ CPUf32{2,2} ] 
```
