# Primitive LU Decomposition

Computes the partial-pivoted LU decomposition of a matrix `operand`:
\$\$P A = L U,\$\$ where \\P\\ is a permutation matrix, \\L\\ is unit
lower triangular, and \\U\\ is upper triangular. `L` (with implicit unit
diagonal) and `U` are packed into a single `LU` output matching LAPACK's
`getrf` layout. \\P\\ is returned in two equivalent forms: `pivots`
(LAPACK's sequential row-swap encoding) and `permutation` (an explicit
permutation vector).

## Usage

``` r
prim_lu(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Matrix of data type floating-point with exactly 2 dimensions.

## Value

`list` of three
[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) values:
`LU` `(m, n)` with the same dtype as the input; `pivots` `(k,)` of dtype
`i32` with `k = min(m, n)` (1-based row swaps such that row `i` was
exchanged with row `pivots[i]` during elimination step `i`); and
`permutation` `(m,)` of dtype `i32`, a 1-based permutation vector for
\\P\\ such that `(P %*% A)[i, ]` equals `A[permutation[i], ]`.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to a `"lu"`
[`stablehlo::hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
(backed by LAPACK on CPU and cuSOLVER on CUDA) for `LU` and `pivots`,
followed by a
[`stablehlo::hlo_while()`](https://r-xla.github.io/stablehlo/reference/hlo_while.html)
loop that converts `pivots` to `permutation` in-graph.

## See also

[`nv_lu()`](https://r-xla.github.io/anvl/reference/nv_lu.md)

## Examples

``` r
x <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
prim_lu(x)
#> $LU
#> AnvlArray
#>   4.0000  6.0000
#>   0.7500 -1.5000
#> [ CPUf64{2,2} ] 
#> 
#> $pivots
#> AnvlArray
#>  1
#>  2
#> [ CPUi32{2} ] 
#> 
#> $permutation
#> AnvlArray
#>  1
#>  2
#> [ CPUi32{2} ] 
#> 
```
