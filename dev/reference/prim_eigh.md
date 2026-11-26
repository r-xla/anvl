# Primitive Symmetric Eigendecomposition

Computes the eigendecomposition of a symmetric matrix `x` of shape
`(n, n)`: \$\$A = \mathrm{vectors} \\ \mathrm{diag}(\mathrm{values}) \\
\mathrm{vectors}^\top.\$\$ Only the lower triangle of `x` is read. The
columns of `vectors` are the (orthonormal) eigenvectors and `values` is
the length-`n` vector of (real) eigenvalues in ascending order. Output
names and order match
[`base::eigen()`](https://rdrr.io/r/base/eigen.html), which unlike this
primitive also handles non-symmetric matrices. The first two axes form
the matrix, and any trailing axes are batch axes.

## Usage

``` r
prim_eigh(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input, with at least 2 axes, the first two forming a symmetric
  square matrix; any trailing axes are batch axes. Can be any float data
  type. An R value materializes at its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

(named `list` of two
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Elements `values` (shape `(n, ...)`) and `vectors` (shape
`(n, n, ...)`), where `(n, n, ...) = shape(x)`. Both have the input's
data type.

## Implemented Rules

- `stablehlo`

- `vectorize`

## StableHLO

Lowers to
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
with target `"eigh"`.

## See also

[`nv_eigh()`](https://r-xla.github.io/anvl/dev/reference/nv_eigh.md)

## Examples

``` r
# `values` and `vectors` both have the input's data type
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
