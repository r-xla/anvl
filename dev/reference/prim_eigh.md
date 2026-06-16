# Primitive Symmetric Eigendecomposition

Computes the eigendecomposition of a symmetric matrix `operand` of shape
`(n, n)`: \$\$A = \mathrm{vectors} \\ \mathrm{diag}(\mathrm{values}) \\
\mathrm{vectors}^\top.\$\$ Only the lower triangle of `operand` is read.
The columns of `vectors` are the (orthonormal) eigenvectors and `values`
is the length-`n` vector of (real) eigenvalues in ascending order.
Output names and order match
[`base::eigen()`](https://rdrr.io/r/base/eigen.html).

## Usage

``` r
prim_eigh(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Symmetric square matrix of floating-point data type.

## Value

Named `list` with elements `values` (length `n`) and `vectors` (shape
`(n, n)`). Both have the same dtype as the input.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
with target `"eigh"`.

## See also

[`nv_eigh()`](https://r-xla.github.io/anvl/dev/reference/nv_eigh.md)

## Examples

``` r
x <- nv_array(c(2, 1, 1, 2), shape = c(2, 2), dtype = "f64")
prim_eigh(x)
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
