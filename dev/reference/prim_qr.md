# Primitive QR Decomposition

Computes the reduced QR decomposition of a matrix `x`: \$\$A = Q R,\$\$
where \\Q\\ has orthonormal columns (\\Q^\top Q = I\\) and \\R\\ is
upper triangular. For an \\m \times n\\ input with \\k = \min(m, n)\\,
\\Q\\ has shape \\m \times k\\ and \\R\\ has shape \\k \times n\\.

## Usage

``` r
prim_qr(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Matrix of data type floating-point with exactly 2 axes.

## Value

Named `list` with elements `Q` (shape `(m, k)`) and `R` (shape
`(k, n)`), where `(m, n) = shape(x)` and `k = min(m, n)`. Both have the
same data type as `x`.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to a `"geqrf"` + `"orgqr"`
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
pair (backed by LAPACK on CPU and cuSOLVER on CUDA) + postprocessing.

## See also

[`nv_qr()`](https://r-xla.github.io/anvl/dev/reference/nv_qr.md)

## Examples

``` r
x <- nv_array(1:6, shape = c(3, 2), dtype = "f32")
prim_qr(x)
#> $Q
#> AnvlArray
#>  -0.2673  0.8729
#>  -0.5345  0.2182
#>  -0.8018 -0.4364
#> [ CPUf32{3,2} ] 
#> 
#> $R
#> AnvlArray
#>  -3.7417 -8.5524
#>   0.0000  1.9640
#> [ CPUf32{2,2} ] 
#> 
```
