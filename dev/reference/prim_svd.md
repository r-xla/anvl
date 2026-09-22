# Primitive Singular Value Decomposition

Computes the reduced ("economy") singular value decomposition of a
matrix `x` of shape `(m, n)`: \$\$A = u \\ \mathrm{diag}(d) \\ vt,\$\$
where `u` has orthonormal columns, `vt` has orthonormal rows, and `d` is
the length-`k` (`k = min(m, n)`) vector of non-negative singular values
in descending order.

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
prim_svd(x)
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
Elements `d` (length `k`), `u` (shape `(m, k)`), and `vt` (shape
`(k, n)`). All have the input's data type.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
with target `"svd"`.

## See also

[`nv_svd()`](https://r-xla.github.io/anvl/dev/reference/nv_svd.md)

## Examples

``` r
# all three outputs have the input's data type
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
