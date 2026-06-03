# Determinant in modulus/sign form

Computes the determinant of a square matrix in the modulus / sign
decomposition matching base R's
[`base::determinant()`](https://rdrr.io/r/base/det.html). For the plain
scalar determinant, use
[`nv_det()`](https://r-xla.github.io/anvl/reference/nv_det.md).

## Usage

``` r
nv_determinant(operand, logarithm = TRUE)

# S3 method for class 'AnvlArray'
determinant(x, logarithm = TRUE, ...)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Square matrix of floating-point data type.

- logarithm:

  (`logical(1)`)  
  If `TRUE` (default), return the log of the absolute determinant.

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Same as `operand`; this is the name used by the base R S3 generic.

- ...:

  No additional arguments.

## Value

Named `list` with elements `modulus` and `sign`, both scalar
[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) with
the same dtype as `operand`. The full determinant is
`sign * exp(modulus)` (with `logarithm = TRUE`) or `sign * modulus`
(with `logarithm = FALSE`).

## Details

For computing the determinant, we use: \$\$P A = L U\$\$ \$\$\det(L) =
1\$\$ \$\$\det(A) = \det(U) / \det(P) = \mathrm{sign}(P^{-1}) \\ \prod_i
U\_{ii} = \mathrm{sign}(P) \\ \prod_i U\_{ii}\$\$

Matching base R's `det_ge_real`, the magnitude is computed in log space
when `logarithm = TRUE` (\\\sum_i \log\|U\_{ii}\|\\) and as a direct
product when `logarithm = FALSE` (\\\prod_i \|U\_{ii}\|\\).

## See also

[`nv_det()`](https://r-xla.github.io/anvl/reference/nv_det.md),
[`nv_solve()`](https://r-xla.github.io/anvl/reference/nv_solve.md),
[`prim_lu()`](https://r-xla.github.io/anvl/reference/prim_lu.md)

## Examples

``` r
a <- nv_matrix(c(4, 3, 6, 3), nrow = 2, dtype = "f64")
nv_determinant(a)
#> $modulus
#> AnvlArray
#>  1.7918
#> [ CPUf64{} ] 
#> 
#> $sign
#> AnvlArray
#>  -1
#> [ CPUf64{} ] 
#> 
nv_determinant(a, logarithm = FALSE)
#> $modulus
#> AnvlArray
#>  6
#> [ CPUf64{} ] 
#> 
#> $sign
#> AnvlArray
#>  -1
#> [ CPUf64{} ] 
#> 
```
