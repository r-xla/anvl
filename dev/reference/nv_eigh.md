# Symmetric Eigendecomposition

Computes the eigendecomposition of a symmetric matrix `x` of shape
`(n, n)`: \$\$A = \mathrm{vectors} \\ \mathrm{diag}(\mathrm{values}) \\
\mathrm{vectors}^\top.\$\$ Only the lower triangle of `x` is read. The
columns of `vectors` are the (orthonormal) eigenvectors and `values` is
the length-`n` vector of (real) eigenvalues in ascending order. Output
names and order match
[`base::eigen()`](https://rdrr.io/r/base/eigen.html), which unlike this
primitive also handles non-symmetric matrices.

## Usage

``` r
nv_eigh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, a symmetric square matrix with exactly 2 axes. Can be any
  numeric data type: a float keeps its own, and an integer one is
  converted to the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and is converted in the same way.

## Value

(named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `values` (length `n`) and `vectors` (shape `(n, n)`). Both have
the input's data type – or the default float data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where the input was an integer one.

## See also

[`prim_eigh()`](https://r-xla.github.io/anvl/dev/reference/prim_eigh.md),
[`base::eigen()`](https://rdrr.io/r/base/eigen.html)

## Examples

``` r
# values and vectors both have the input's data type
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

# an integer matrix is decomposed at the default float data type
nv_eigh(nv_matrix(c(2L, 1L, 1L, 2L), nrow = 2))
#> $values
#> AnvlArray
#>  1
#>  3
#> [ CPUf32{2} ] 
#> 
#> $vectors
#> AnvlArray
#>  -0.7071  0.7071
#>   0.7071  0.7071
#> [ CPUf32{2,2} ] 
#> 
```
