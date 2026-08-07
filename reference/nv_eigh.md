# Symmetric Eigendecomposition

Computes the eigendecomposition of a symmetric matrix `x` of shape
`(n, n)`: \$\$A = \mathrm{vectors} \\ \mathrm{diag}(\mathrm{values}) \\
\mathrm{vectors}^\top.\$\$ Only the lower triangle of `x` is read. The
columns of `vectors` are the (orthonormal) eigenvectors and `values` is
the length-`n` vector of (real) eigenvalues in ascending order. Output
names and order match
[`base::eigen()`](https://rdrr.io/r/base/eigen.html).

## Usage

``` r
nv_eigh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Symmetric square matrix of floating-point data type.

## Value

Named `list` with elements `values` (length `n`) and `vectors` (shape
`(n, n)`). Both have the same dtype as the input.

## See also

[`prim_eigh()`](https://r-xla.github.io/anvl/reference/prim_eigh.md),
[`base::eigen()`](https://rdrr.io/r/base/eigen.html)

## Examples

``` r
x <- nv_matrix(c(2, 1, 1, 2), nrow = 2, dtype = "f64")
nv_eigh(x)
#> $values
#> AnvlArray
#>  1
#>  3
#> [ CPUf64{2} ] 
#> 
#> $vectors
#> AnvlArray
#>  -0.7071  0.7071
#>   0.7071  0.7071
#> [ CPUf64{2,2} ] 
#> 
```
