# Primitive LU Decomposition

Computes the partial-pivoted LU decomposition of a matrix `x`: \$\$P A =
L U,\$\$ where \\P\\ is a permutation matrix, \\L\\ is unit lower
triangular, and \\U\\ is upper triangular. `L` (with implicit unit
diagonal) and `U` are packed into a single `LU` output matching LAPACK's
`getrf` layout. \\P\\ is returned in two equivalent forms: `pivots`
(LAPACK's sequential row-swap encoding) and `permutation` (an explicit
permutation vector).

## Usage

``` r
prim_lu(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with exactly 2 axes. Can be any float data type. An R value
  materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

(named `list` of three
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `LU` `(m, n)` with the input's data type; `pivots` `(k,)` at
the default integer data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
with `k = min(m, n)` (sequential row swaps: row `i` was exchanged with
row `pivots[i]` during elimination step `i`); and `permutation` `(m,)`
at that same data type, a permutation vector for \\P\\ such that
`(P %*% A)[i, ]` equals `A[permutation[i], ]`.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to a `"lu"`
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
(backed by LAPACK on CPU and cuSOLVER on CUDA) for `LU` and `pivots`,
followed by a
[`hlo_while()`](https://r-xla.github.io/stablehlo/reference/hlo_while.html)
loop that converts `pivots` to `permutation` in-graph.

## See also

[`nv_lu()`](https://r-xla.github.io/anvl/dev/reference/nv_lu.md)

## Examples

``` r
# `LU` keeps the input's data type; the pivots are the default integer
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
