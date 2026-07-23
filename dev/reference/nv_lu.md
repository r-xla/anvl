# LU Decomposition

Computes the partial-pivoted LU decomposition of a matrix `x`: \$\$P A =
L U,\$\$ where \\P\\ is a permutation matrix, \\L\\ is unit lower
triangular, and \\U\\ is upper triangular.

This function returns `L` and `U` as separate matrices. Use
[`prim_lu()`](https://r-xla.github.io/anvl/dev/reference/prim_lu.md) to
get them in packed `LU` form.

## Usage

``` r
nv_lu(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Matrix of data type floating-point with exactly 2 axes.

## Value

Named `list`:

- `L` – unit lower-triangular factor of shape `(m, k)`, where
  `(m, n) = shape(x)` and `k = min(m, n)`.

- `U` – upper-triangular factor of shape `(k, n)`.

- `pivots` – length `k`, dtype `i32`. LAPACK-style sequential 1-based
  row swaps as returned by `getrf`.

- `permutation` – length `m`, dtype `i32`. A 1-based permutation vector
  representing \\P\\.

## See also

[`prim_lu()`](https://r-xla.github.io/anvl/dev/reference/prim_lu.md)

## Examples

``` r
x <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
nv_lu(x)
#> $L
#> AnvlArray
#>  1.0000 0.0000
#>  0.7500 1.0000
#> [ CPUf64{2,2} ] 
#> 
#> $U
#> AnvlArray
#>   4.0000  6.0000
#>   0.0000 -1.5000
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
