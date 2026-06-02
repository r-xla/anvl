# Primitive Singular Value Decomposition

Computes the reduced ("economy") singular value decomposition of a
matrix `operand` of shape `(m, n)`: \$\$A = u \\ \mathrm{diag}(d) \\
vt,\$\$ where `u` has orthonormal columns, `vt` has orthonormal rows,
and `d` is the length-`k` (`k = min(m, n)`) vector of non-negative
singular values in descending order.

Note: unlike [`base::svd()`](https://rdrr.io/r/base/svd.html), which
returns the right singular vectors as `v` of shape `(n, k)` (so that
`a = u %*% diag(d) %*% t(v)`), this primitive returns them already
transposed as `vt` of shape `(k, n)` (matching the underlying LAPACK /
cuSOLVER output and avoiding an extra transpose).

Supports any matrix shape on both the host (LAPACK `gesdd`) and CUDA
(cuSOLVER `gesvd`) backends. cuSOLVER's `m >= n` requirement is handled
transparently via a layout flip for wide matrices.

## Usage

``` r
prim_svd(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Matrix of data type floating-point with exactly 2 dimensions.

## Value

Named `list` with elements `d` (length `k`), `u` (shape `(m, k)`), and
`vt` (shape `(k, n)`). All have the same dtype as the input.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`stablehlo::hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
with target `"svd"`.

## See also

[`nv_svd()`](https://r-xla.github.io/anvl/dev/reference/nv_svd.md)

## Examples

``` r
x <- nv_array(c(1, 0, 0, 1, 0, 1), shape = c(3, 2))
prim_svd(x)
#> $d
#> AnvlArray
#>  1.6180
#>  0.6180
#> [ CPUf32{2} ] 
#> 
#> $u
#> AnvlArray
#>   0.8507  0.5257
#>   0.0000  0.0000
#>   0.5257 -0.8507
#> [ CPUf32{3,2} ] 
#> 
#> $vt
#> AnvlArray
#>   0.5257  0.8507
#>   0.8507 -0.5257
#> [ CPUf32{2,2} ] 
#> 
```
